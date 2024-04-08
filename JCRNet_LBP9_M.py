import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.ops import functional as F

from ipdb import set_trace as stxx
import ipdb

##########################################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, pad_mode='pad',
        padding=(kernel_size // 2), has_bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, has_bias=bias, pad_mode='pad')


class ResBlock(nn.Cell):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):      ###############从0到2，不包含2，即0,1
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(*m)
        self.res_scale = res_scale

    def construct(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        # stxx()
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Cell):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.SequentialCell(*modules_body)

    def construct(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

    
class CALayer(nn.Cell):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.SequentialCell(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, has_bias=bias),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, has_bias=bias),
            nn.Sigmoid()
        )

    def construct(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 

    
##########################################################################
## Compute inter-stage features
class SAM(nn.Cell):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def construct(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img    #feats  image


class mergeblock(nn.Cell):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def construct(self, x, bridge):
        out = ops.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_*w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = ops.matmul(V_t, V)
        mat_inv = ops.inverse(mat)
        project_mat = ops.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_*w_)
        project_feature = ops.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = ops.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = ops.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out+x

    
##########################################################################
## U-Net    
class Encoder(nn.Cell):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff,depth=3):
        super(Encoder, self).__init__()
        self.body=nn.CellList()#[]
        self.depth=depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*i, out_size=n_feat+scale_unetfeats*(i+1), downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True))
        self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*(depth-1), out_size=n_feat+scale_unetfeats*(depth-1), downsample=False, relu_slope=0.2, use_csff=csff, use_HIN=True))

    def construct(self, x, encoder_outs=None, decoder_outs=None):
        stxx()
        res=[]
        if encoder_outs is not None and decoder_outs is not None:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x,encoder_outs[i],decoder_outs[-i-1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res,x

    
class UNetConvBlock(nn.Cell):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, kernel_size=1, stride = 1, padding = 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3,pad_mode='pad', padding=1, has_bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, )
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3,pad_mode='pad',padding=1, has_bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, kernel_size=3,stride =1, padding=1, pad_mode='pad')
            self.csff_dec = nn.Conv2d(in_size, out_size, kernel_size=3,stride =1, padding=1, pad_mode='pad')
            self.phi = nn.Conv2d(out_size, out_size, kernel_size=3,stride =1, padding=1, pad_mode='pad')
            self.gamma = nn.Conv2d(out_size, out_size,kernel_size=3,stride =1, padding=1, pad_mode='pad')

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def construct(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = ops.chunk(out, 2, axis=1)
            out = ops.cat([self.norm(out_1), out_2], axis=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1)
            out = out*F.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Cell):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Conv2dTranspose(in_size, out_size, kernel_size=2, stride=2, has_bias=True)
        self.conv_block = UNetConvBlock(out_size*2, out_size, False, relu_slope)

    def construct(self, x, bridge):
        up = self.up(x)
        out = ops.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
    

class Decoder(nn.Cell):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=3):
        super(Decoder, self).__init__()
        
        self.body=nn.CellList()
        self.skip_conv=nn.CellList()#[]
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat+scale_unetfeats*(depth-i-1), out_size=n_feat+scale_unetfeats*(depth-i-2), relu_slope=0.2))
            self.skip_conv.append(nn.Conv2d(n_feat+scale_unetfeats*(depth-i-1), n_feat+scale_unetfeats*(depth-i-2),kernel_size=3,stride =1, padding=1, pad_mode='pad'))
         
    def construct(self, x, bridges):
        res=[]
        for i,up in enumerate(self.body):
            x=up(x,self.skip_conv[i](bridges[-i-1]))
            res.append(x)

        return res


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Cell):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.SequentialCell(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=1, stride=1, padding=0, has_bias=False))

    def construct(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.SequentialCell(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=False))

    def forward(self, x):
        x = self.up(x)
        return x



#################################反投影理论LBP#########################################
class LBP(nn.Cell):
    def __init__(self, input_size=3, output_size=3, kernel_size=3, stride=1, padding=1):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size,output_size)
        self.conv1_1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)   #64
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)    #64
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)    #64
        self.local_weight1_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True) #64
        self.local_weight2_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True) #64

    def construct(self, x):
        x=self.fusion(x)       #低光图像通过特征融合结构，相当于通过一个注意力
        hr = self.conv1_1(x)   #亮操作  得到  亮图像
        lr = self.conv2(hr)    #暗操作  得到  暗图像
        residue = self.local_weight1_1(x) - lr   #低光图像 减去 暗图像  得到 低暗残差图
        h_residue = self.conv3(residue)    #对低暗残差图 进行 亮操作
        # hr_weight = self.local_weight2_1(hr)  #亮图像
        # return hr_weight + h_residue
        return h_residue

class LightenBlock(nn.Cell):
    def __init__(self, input_size=64, output_size=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(LightenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def construct(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code+offset
        out = self.conv_Decoder(code_lighten)
        return out

class DarkenBlock(nn.Cell):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def construct(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code-offset
        out = self.conv_Decoder(code_lighten)
        return out

class FusionLayer(nn.Cell):
    def __init__(self, inchannel, outchannel, reduction=3):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(inchannel, inchannel // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(inchannel // reduction, inchannel, has_bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def construct(self, x):
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y

class ConvBlock(nn.Cell):
    def __init__(self, input_size=64, output_size=64, kernel_size=3, stride=1, padding=1, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride = stride, padding = padding, has_bias=bias,pad_mode='pad')
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = nn.PReLU()

    def construct(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out

##########################################################################
##########################illuminationNet
##########################################################################
class SFTLayer(nn.Cell):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, kernel_size=1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, kernel_size=1)
        self.SFT2 = nn.Conv2d(128, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')

    def construct(self, x,c):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(c), 0.1))   #scale输出为64
        m = F.leaky_relu(self.SFT2(ops.cat([x, scale], 1)))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(c), 0.1))
        return m+shift    #输出64


class ResBlock_SFT(nn.Cell):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')

    def construct(self, x,c):    #x通道数64  c通道数32
        fea = self.sft0(x,c)
        fea = F.relu(self.conv0(fea))
        fea = self.sft1(fea, c)
        fea = self.conv1(fea)   #输入64  输出64
        return x + fea     #输出64

class illuminationNet(nn.Cell):
    def __init__(self):
        super(illuminationNet, self).__init__()

        self.conv11 = nn.Conv2d(3, 32, 3, stride=1, padding=1,pad_mode='pad')
        self.conv12 = nn.Conv2d(32, 32, 3, stride=1, padding=1,pad_mode='pad')
        self.conv13 = nn.Conv2d(32, 64, 3, stride=1, padding=1,pad_mode='pad')
        self.conv14 = nn.Conv2d(64, 64, 3, stride=1, padding=1,pad_mode='pad')
        self.conv15 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.sft = ResBlock_SFT()

        self.CondNet = nn.SequentialCell(
            nn.Conv2d(1, 64, 3, stride=1, padding=1,pad_mode='pad'), nn.ReLU(), nn.Conv2d(64, 64, 1),
            nn.ReLU(), nn.Conv2d(64, 32, 1))

        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')

        # self.opt = opt
        # self.skip = skip

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn9 = nn.BatchNorm2d(128)
        self.relu9 = nn.ReLU()

        self.conv25 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1,pad_mode='pad')
        self.bn25 = nn.BatchNorm2d(128)
        self.relu25 = nn.ReLU()
        # 26
        self.conv26 = nn.Conv2d(128, 3, kernel_size=1, stride = 1, padding = 0)
        self.bn26 = nn.BatchNorm2d(3)

        # if self.opt.tanh:
        self.tanh = nn.Sigmoid()

    def construct(self, input):
        input = Tensor(input)
        edge = self.relu(self.conv11(input))   #输入3  输出32
        edge = self.relu(self.conv12(edge))
        edge = self.relu(self.conv13(edge))
        edge = self.relu(self.conv14(edge))
        edge = self.relu(self.conv15(edge))    #输入64  输出1

        input_fea = self.conv0(input)
        edge_fea = self.CondNet(edge)

        x = self.sft(input_fea,edge_fea)

        x = self.relu3(self.bn3(self.conv3(x)))

        res1 = x

        x = self.bn4(self.conv4(x))
        x = self.relu4(x)

        x = self.bn5(self.conv5(x))
        x = self.relu5(x + res1)

        x = self.bn8(self.conv8(x))
        x = self.relu8(x)

        x = self.bn9(self.conv9(x))
        x = self.relu9(x)
        res7 = x

        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(x)      #128  3

        # if self.opt.tanh:
        latent = self.tanh(latent)
        # if self.opt.tanh:
        #     latent = self.tanh(latent)           #输入128  输出3
        # output = input / (latent + 0.00001)
        # return latent, output, edge
        return latent

##########################################################################

##########################################################################
## JCRNet
class JCRNet(nn.Cell):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, depth=1):
        super(JCRNet, self).__init__()

        act = nn.PReLU()
        self.depth=depth
        # self.basic=Basic_block(in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats, num_cab, kernel_size, reduction, bias)
        self.shallow_feat1 = nn.SequentialCell(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat7 = nn.SequentialCell(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=2, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=2)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.phi_0 = ResBlock(default_conv,3,3)
        self.phit_0 = ResBlock(default_conv,3,3)
        self.phi_6 = ResBlock(default_conv,3,3)
        self.phit_6 = ResBlock(default_conv,3,3)
        self.r0 = Tensor([0.4412])
        self.r6 = Tensor([0.4965])

        self.concat67 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, 3, kernel_size, bias=bias)
        self.lbp = LBP(input_size=3, output_size=3, kernel_size=3, stride=1, padding=1)
        self.conv40_64 = nn.SequentialCell(
            nn.Conv2d(40, 64, kernel_size=3, stride = 1, padding = 1,pad_mode='pad'),
            nn.ReLU()
        )

        self.conv64_40 = nn.SequentialCell(
            nn.Conv2d(64, 40, kernel_size=3, stride = 1, padding = 1,pad_mode='pad'),
            nn.ReLU()
        )
        self.illum = illuminationNet()

    def construct(self, img):
        stxx()
        res=[]
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_1 = self.phi_0(img) - img
        x1_img = img - self.r0*self.phit_0(phixsy_1)
        ## PMM
        x1 = self.shallow_feat1(x1_img)
x
        feat1,feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1,feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)
        res.append(stage1_img)

        ##-------------------------------------------
        ##-------------- Stage 2-6 ---------------------
        ##-------------------------------------------
        # for _ in range(self.depth):
        #     x2_samfeats, stage1_img, feat1, res1 = self.basic(img,stage1_img,feat1,res1,x2_samfeats)
        #     res.append(stage1_img)

        ####################细化增强网络####################
        stage1_img = self.illum(stage1_img)
        ####################细化增强网络####################

        ##-------------------------------------------
        ##-------------- Stage 7---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_7 = self.phi_6(stage1_img) - img
        x7_img = stage1_img - self.r6*self.phit_6(phixsy_7)
        x7 = self.shallow_feat7(x7_img)
        ## PMM
        x7_cat = self.concat67(ops.cat([x7, x2_samfeats], 1))
        stage7_img = self.tail(x7_cat)+ img
        lbp = self.lbp(img)    #原始图像的反投影残差
        stage7_img = stage7_img + lbp
        res.append(stage7_img)

        return res[::-1]
