import torch
from torch import nn
import torch.nn.functional as F
from loss import batch_episym

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx[:, :, :]

class attention_propagantion(nn.Module):

    def __init__(self,channel,head,k):
        nn.Module.__init__(self)
        self.head=head
        self.head_dim=channel//head
        self.knn = k
        self.query_filter,self.key_filter,self.value_filter=nn.Conv1d(channel,channel,kernel_size=1),nn.Conv1d(channel,channel,kernel_size=1),\
                                                            nn.Conv1d(channel,channel,kernel_size=1)
        self.mh_filter=nn.Conv1d(channel,channel,kernel_size=1)
        self.cat_filter=nn.Sequential(nn.Conv1d(2*channel,2*channel, kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                      nn.Conv1d(2*channel, channel, kernel_size=1))

    def forward(self,desc1):
        desc1 = desc1.squeeze(-1)
        batch_size=desc1.shape[0]
        query,key,value=self.query_filter(desc1).view(batch_size,self.head,self.head_dim,-1),self.key_filter(desc1).view(batch_size,self.head,self.head_dim,-1),\
                        self.value_filter(desc1).view(batch_size,self.head,self.head_dim,-1)
        
        # BNK -> B4NK
        idx = knn(desc1, self.knn).unsqueeze(1).repeat(1,4,1,1)
        score = torch.einsum('bhdn,bhdm->bhnm',query,key)/ self.head_dim ** 0.5
        mask = torch.zeros_like(score).scatter_(-1, idx, torch.ones_like(idx, dtype=score.dtype))
        # mask: B4NN
        mask = torch.mul(mask, mask.transpose(-1,-2))
        score = torch.mul(score, mask) + -1e9 * (1 - mask)
        score = torch.softmax(score, dim=-1)
        
        add_value=torch.einsum('bhnm,bhdm->bhdn',score,value).reshape(batch_size,self.head_dim*self.head,-1)
        add_value=self.mh_filter(add_value)
        desc1_new=desc1+self.cat_filter(torch.cat([desc1,add_value],dim=1))
        desc1_new = desc1_new.unsqueeze(-1)
        return desc1_new

class PointCA(nn.Module):
    def __init__(self, channel, reduction=2):
        super(PointCA, self).__init__()
        inter_channel = int(channel // reduction)
        self.seed_predictor = nn.Sequential(
                nn.InstanceNorm2d(channel, eps=1e-3),
                nn.SyncBatchNorm(channel),
                nn.ReLU(),
                nn.Conv2d(channel, 1, kernel_size=1),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(channel, inter_channel, kernel_size=1),
            nn.SyncBatchNorm(inter_channel),
            nn.ReLU(),
            nn.Conv2d(inter_channel, channel, kernel_size=1),
        )
        
        
    def forward(self, x):
        # b = x.size(0)
        w = self.seed_predictor(x) # b*1*n*1
        w = torch.tanh(torch.relu(w))
        w = F.normalize(w, p=1, dim=2)
        
        x_w = torch.mul(x, w.expand_as(x)) # b*c*n*1
        x_sum =torch.sum(x_w,dim=2, keepdim=True) # bc11
         
        out = self.conv(x_sum)
        out = torch.sigmoid(out)
        out = out * x # bc11*bcn1
        return out
    
class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.SyncBatchNorm(out_channels),
                nn.ReLU(),
                PointCA(out_channels, 4),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.SyncBatchNorm(out_channels),
                nn.ReLU(),
                )
    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
    
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.SyncBatchNorm(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.SyncBatchNorm(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.SyncBatchNorm(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.SyncBatchNorm(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)# b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.SyncBatchNorm(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):
        #x_up: b*c*n*1
        #x_down: b*c*k*1
        embed = self.conv(x_up)# b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out



class RNABlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        self.l1_1.append(PointCN(channels))
        self.l1_1.append(PointCN(channels))
        self.l1_1.append(PointCN(channels))
        self.l1_1.append(attention_propagantion(channels, 4, 40))
        
        

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        
        self.l1_2.append(nn.Sequential(nn.Conv2d(2*channels, channels, kernel_size=1),
                                       nn.SyncBatchNorm(channels),
                                       nn.ReLU(),
                                       ))
        self.l1_2.append(attention_propagantion(channels, 4, 40))
        
        self.l1_2.append(PointCN(channels))
        self.l1_2.append(PointCN(channels))
        self.l1_2.append(PointCN(channels))
        

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs):
        #data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2( torch.cat([x1_1,x_up], dim=1))

        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class RNAGNN(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = RNABlock(config.net_channels, 4+self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [RNABlock(config.net_channels, 6+self.side_channel, depth_each_stage, config.clusters) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        #data: b*1*n*c
        input = data['xs'].transpose(1,3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat  


        
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat