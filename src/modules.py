import pytorch_lightning as pl

class ConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, selfcon_pos=[False,False,False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(ConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(selfcon_pos=selfcon_pos, selfcon_arch=selfcon_arch, selfcon_size=selfcon_size, dataset=dataset)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
            
            self.sub_heads = []
            for pos in selfcon_pos:
                if pos:
                    self.sub_heads.append(nn.Linear(dim_in, feat_dim))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            
            heads = []
            for pos in selfcon_pos:
                if pos:
                    heads.append(nn.Sequential(
                        nn.Linear(dim_in, dim_in),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim_in, feat_dim)
                    ))
            self.sub_heads = nn.ModuleList(heads)
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        sub_feat, feat = self.encoder(x)
        
        sh_feat = []
        for sf, sub_head in zip(sub_feat, self.sub_heads):
            sh_feat.append(F.normalize(sub_head(sf), dim=1))
        
        feat = F.normalize(self.head(feat), dim=1)
        return sh_feat, feat
    
    
class CEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', method='ce', num_classes=10, dim_out=128, selfcon_pos=[False,False,False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(CEResNet, self).__init__()
        self.method = method
        
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(selfcon_pos=selfcon_pos, selfcon_arch=selfcon_arch, selfcon_size=selfcon_size, dataset=dataset)
        
        logit_fcs, feat_fcs = [], []
        for pos in selfcon_pos:
            if pos:
                logit_fcs.append(nn.Linear(dim_in, num_classes))
                feat_fcs.append(nn.Linear(dim_in, dim_out))
                
        self.logit_fc = nn.ModuleList(logit_fcs)
        self.l_fc = nn.Linear(dim_in, num_classes)
        
        if method not in ['ce', 'subnet_ce', 'kd']:
            self.feat_fc = nn.ModuleList(feat_fcs)
            self.f_fc = nn.Linear(dim_in, dim_out)
            for param in self.f_fc.parameters():
                param.requires_grad = False

    def forward(self, x):
        sub_feat, feat = self.encoder(x)
        
        feats, logits = [], []
        
        for idx, sh_feat in enumerate(sub_feat):
            logits.append(self.logit_fc[idx](sh_feat))
            if self.method not in ['ce', 'subnet_ce', 'kd']:
                out = self.feat_fc[idx](sh_feat)
                feats.append(F.normalize(out, dim=1))
            
        if self.method not in ['ce', 'subnet_ce', 'kd']:
            return [feats, F.normalize(self.f_fc(feat), dim=1)], [logits, self.l_fc(feat)]
        else:
            return [logits, self.l_fc(feat)]