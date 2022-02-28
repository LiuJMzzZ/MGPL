import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import net


class MGPLnet(nn.Module):
    def __init__(self, arch='resnet18', channel=3, latent_dim=128, n_sub_prototypes=3, n_classes=10, temp_inter=0.1, temp_intra=1, init=True):
        super(MGPLnet, self).__init__()
        self.arch = arch
        self.channel = channel
        self.latent_dim = latent_dim
        self.n_sub_prototypes = n_sub_prototypes
        self.n_classes = n_classes
        self.temp_inter = temp_inter
        self.temp_intra = temp_intra
        self.encoder, self.decoder = net(self.arch, self.channel, self.latent_dim)
        self.prototypes = nn.Parameter(torch.randn(self.n_classes*self.n_sub_prototypes, self.latent_dim).cuda(), requires_grad=True) 
        # self.fc = nn.Linear(self.latent_dim, self.n_classes) 
        if init:
            nn.init.kaiming_normal_(self.prototypes)

    def sampler(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = epsilon * std + mu
        else:
            z = mu
        return z

    def distance(self, latent_z, prototypes):
        '''
        Compute the squared Euclidean (L2_Norm) distance from instances in a batch to all prototypes.

        Input:
        matrixA: [batch_size, latent_dim], instance features in one batch 
        matrixB: [prototype_num, latent_dim], all prototypes features, where prototype_num = n_classes * n_sub_prots

        Output:
        distance_matrix: tensor [batch_size, prototype_num], the squared Euclidean distance (L2_norm) 
        '''
        matrixA, matrixB = latent_z, prototypes 
        matrixB_t = matrixB.transpose(0,1)
        matrixA_square = torch.sum(torch.pow(matrixA,2),1, keepdim=True)
        matrixB_square = torch.sum(torch.pow(matrixB_t,2),0, keepdim=True)
        product_A_B = torch.matmul(matrixA, matrixB_t)
        distance_matrix = matrixA_square + matrixB_square - 2 * product_A_B
        return distance_matrix

    def kl_div_to_prototypes(self, mean, logvar, prototypes):
        '''
        Compute the KL divergence between z~N(mu, sigma) to all prototypes w~N(mu_w, I)
        Note that kl div sum over all z dimentions, result that the weight of kl term increase with z_dim

        Input 
        mean, var: The output from Encoder, shape [batch_size, latent_dim]
        prototypes: all prototypes, shape [prototype_num, latent_dim]
        '''
        kl_div = self.distance(mean, prototypes) + torch.sum((logvar.exp() - logvar - 1), axis=1, keepdims=True)
        return 0.5 * kl_div

    def forward(self, x):
        mu, logvar, lateral_z = self.encoder(x)
        latent_z = self.sampler(mu, logvar)

        # L2 distance from z to all prototypes
        dist = self.distance(latent_z, self.prototypes) 

        # KL divergence from z to all prototypes
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes) 

        recon_x = self.decoder(latent_z, lateral_z)
        return latent_z, dist, kl_div, recon_x
        
    def loss(self, x, y):
        latent_z, dist, kl_div, x_recon = self.forward(x)

        # prediction by the nearest prototype
        dist_reshape = dist.reshape(len(x), self.n_classes, self.n_sub_prototypes)  
        dist_class_min, _ = torch.min(dist_reshape, axis=2)  # min distance to the sub_prototypes of each class 
        _, preds = torch.min(dist_class_min, axis=1)  # class label prediction by the minimum distance
        
        # compute feature distance to the prototypes of the ground-truth classes
        y_one_hot = F.one_hot(y, num_classes=self.n_classes) 
        y_mask = y_one_hot.repeat_interleave(self.n_sub_prototypes,dim=1).bool()  # mask the dist with real label y 
        dist_y = dist[y_mask].reshape(len(dist), self.n_sub_prototypes)  # dist to w_y (prototypes with label y)
        kl_div_y = kl_div[y_mask].reshape(len(kl_div), self.n_sub_prototypes) # KLD to w_y (prototypes with label y)
        q_w_z_y = F.softmax(-dist_y / self.temp_intra, dim=1)  # q(w|z,y) 
        
        ############ Generative Constraint ############
        # recon_loss
        rec_loss = F.binary_cross_entropy(x_recon, x)  
        # rec_loss = F.mse_loss(x_recon, x)
        # prototypes conditional prior
        kld_loss = torch.mean(torch.sum(q_w_z_y * kl_div_y, dim=1)) 
        # entropy prior
        ent_loss = torch.mean(torch.sum(q_w_z_y * torch.log(q_w_z_y * self.n_sub_prototypes + 1e-7), dim=1))

        ############ Distriminative Constraint ############
        # using logsumexp(LSE) for numerically stabilized computation 
        LSE_all_dist = torch.logsumexp(-dist / self.temp_inter, 1)
        LSE_target_dist = torch.logsumexp(-dist_y / self.temp_inter, 1)
        dis_loss = torch.mean(LSE_all_dist - LSE_target_dist)

        loss = {'dis': dis_loss, 'rec': rec_loss, 'kld': kld_loss, 'ent': ent_loss,}
        return latent_z, x_recon, preds, loss


def train(MGPLnet, args, train_loader, epoch, optimizer):
    num_epochs = args.epoch
    MGPLnet.train()
    print('Current learning rate is {}'.format(optimizer.param_groups[0]['lr']))
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('*' * 70)
        
    train_corrects = 0
    running_loss = {}

    for i, (image, label) in enumerate(train_loader):
        image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        _, _, preds, loss = MGPLnet.loss(image, label)
        total_loss =  args.lamda * (loss['rec'] + loss['kld'] +  loss['ent']) + (1 - args.lamda) * loss['dis']
        loss['total'] = total_loss
        total_loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(MGPLnet.parameters(), max_norm=10)
        optimizer.step()
        
        running_loss = {k: loss.get(k, 0).item() + running_loss.get(k, 0) for k in loss.keys()}
        train_corrects += torch.sum(preds == label.data)

    train_acc = train_corrects.item() / len(train_loader.dataset)
    train_loss = {k: running_loss.get(k, 0) / len(train_loader) for k in running_loss.keys()}
    
    print('Train corrects: {} Train samples: {} Train accuracy: {}'.format(
        train_corrects, len(train_loader.dataset), train_acc))
    print('Train loss: {:.3f}= {}*[rec({:.3f}) + kld({:.3f}) + ent({:.3f})] + (1-{})*dis({:.3f})'.format(
            train_loss['total'], args.lamda, train_loss['rec'], train_loss['kld'],
            train_loss['ent'],  args.lamda, train_loss['dis']))
    

def val(MGPLnet, args, val_loader, epoch):
    MGPLnet.eval()

    val_corrects = 0.0
    val_running_loss = {}
    
    for image, label in val_loader:
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            latent_z, x_recon, preds, loss = MGPLnet.loss(image, label)
            total_loss =  args.lamda * (loss['rec'] + loss['kld'] +  loss['ent']) + (1 - args.lamda) * loss['dis']
            loss['total'] = total_loss
            val_running_loss = {k: loss.get(k, 0).item() + val_running_loss.get(k, 0) for k in loss.keys()}
            val_corrects += torch.sum(preds == label.data)

    val_acc = val_corrects.item() / len(val_loader.dataset)
    val_loss = {k: val_running_loss.get(k, 0) / len(val_loader) for k in val_running_loss.keys()}
 
    print('Val corrects: {} Val samples: {} Val accuracy: {}'.format(
        val_corrects, len(val_loader.dataset), val_acc))
    print('Val loss: {:.3f}= {}*[rec({:.3f}) + kld({:.3f}) + ent({:.3f})] + (1-{})*dis({:.3f})'.format(
            val_loss['total'], args.lamda, val_loss['rec'], val_loss['kld'],
            val_loss['ent'],  args.lamda, val_loss['dis']))

    print('*' * 70)
    
    return val_acc


if __name__ == '__main__':
    net = MGPLnet() 
    print(net.parameters)