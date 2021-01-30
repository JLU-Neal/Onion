# pylint: disable=E1101,R,C
import torch.nn as nn
import torch.nn.functional as F
import torch
from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate, \
    setup_so3_integrate

from PointNet.pointnet import PointNetEncoder, STN3d, STNkd
from PointNet.pointnet_cls import get_model
from Blocks import SphericalBlock, PointNetBlock, SO3Block


class Model(nn.Module):
    def __init__(self, nclasses):
        super().__init__()

        self.features = [3, 128, nclasses]
        self.bandwidths = [24, 24]

        # define the number of layers.
        self.num_layers = 1

        assert len(self.bandwidths) == len(self.features) - 1

        sequence = []

        # S2 layer
        # =====================If we could chose another gridding method=====================
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2Convolution(self.features[0], int(self.features[1] / self.num_layers), self.bandwidths[0],
                                      self.bandwidths[1], grid))
        # sequence.append(S2Convolution(self.features[0], int(self.features[2] / self.num_layers), self.bandwidths[1],
        #                               self.bandwidths[2], grid))

        # SO3 layers
        # for l in range(1, len(self.features) - 2):
        #     nfeature_in = self.features[l]
        #     nfeature_out = self.features[l + 1]
        #     b_in = self.bandwidths[l]
        #     b_out = self.bandwidths[l + 1]
        #
        #     sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
        #     sequence.append(nn.ReLU())+
        #     grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
        #     sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        sequence.append(nn.BatchNorm3d(self.features[-2], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

        # Output layer
        output_features = self.features[-2]
        # self.out_layer = nn.Linear(2*output_features, self.features[-1]) #For ensemble learning
        self.out_layer_1 = nn.Linear(self.features[-2], self.features[-2])
        self.out_layer = nn.Linear(self.features[-2], self.features[-1])
        self.bn1 = nn.BatchNorm1d(self.features[-2])
        self.bn2 = nn.BatchNorm1d(self.features[-2])
        self.relu = nn.LeakyReLU()

        # self.point_net = PointNetEncoder(global_feat = False)
        self.point_net = get_model(combine_with_spherical_features=True)
        print("fuck")

    def change_coordinates(self, coords, p_from='C', p_to='S'):
        """
        Change Spherical to Cartesian coordinates and vice versa, for points x in S^2.

        In the spherical system, we have coordinates beta and alpha,
        where beta in [0, pi] and alpha in [0, 2pi]

        We use the names beta and alpha for compatibility with the SO(3) code (S^2 being a quotient SO(3)/SO(2)).
        Many sources, like wikipedia use theta=beta and phi=alpha.

        :param coords: coordinate array
        :param p_from: 'C' for Cartesian or 'S' for spherical coordinates
        :param p_to: 'C' for Cartesian or 'S' for spherical coordinates
        :return: new coordinates
        """
        if p_from == p_to:
            return coords
        elif p_from == 'S' and p_to == 'C':

            beta = coords[..., 0]
            alpha = coords[..., 1]
            r = 1.

            out = torch.empty(beta.shape + (3,))

            ct = torch.cos(beta)
            cp = torch.cos(alpha)
            st = torch.sin(beta)
            sp = torch.sin(alpha)
            out[..., 0] = r * st * cp  # x
            out[..., 1] = r * st * sp  # y
            out[..., 2] = r * ct  # z
            return out

        elif p_from == 'C' and p_to == 'S':

            x = coords[..., 0]
            y = coords[..., 1]
            z = coords[..., 2]

            out = torch.empty(x.shape + (2,))
            out[..., 0] = torch.acos(z)  # beta
            out[..., 1] = torch.atan2(y, x)  # alpha
            return out

        else:
            raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))

    def devoxelize(self, points, bandwidth):
        # points = points.cuda()
        try:
            x = points[:, 0]
        except IndexError:
            print(points.shape[0], points.shape[1])

        y = points[:, 1]
        z = points[:, 2]

        # Fistly, we get the centroid, then translate the points
        centroid_x = torch.sum(x) / points.shape[0]
        centroid_y = torch.sum(y) / points.shape[0]
        centroid_z = torch.sum(z) / points.shape[0]
        centroid = torch.tensor([centroid_x, centroid_y, centroid_z])
        points = points.float()
        points -= centroid.cuda()

        # After normalization, compute the distance between the sphere and points

        radius = torch.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2 + points[..., 2] ** 2)

        # print(points.size())
        # print(radius.size())

        radius = radius.unsqueeze(2)
        radius = radius.repeat(1, 1, 3)
        points_on_sphere = points / radius
        # ssgrid = sgrid.reshape(-1, 3)
        # phi, theta = S2.change_coordinates(ssgrid, p_from='C', p_to='S')
        out = self.change_coordinates(points_on_sphere, p_from='C', p_to='S')

        phi = torch.tensor(out[..., 0]).cuda()
        theta = torch.tensor(out[..., 1]).cuda()
        pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
        phi = phi
        theta = theta % (pi * 2)

        b = bandwidth  # bandwidth
        # By computing the m,n, we can find
        # the neighbours on the sphere
        m = torch.trunc((phi - pi / (4 * b)) / (pi / (2 * b)))
        m = m.long()
        n = torch.trunc(theta / (pi / b))
        n = n.long()
        # need to mind the boundary issues
        m_boundary = m >= 2 * b
        n_boundary = n >= 2 * b
        m[m_boundary] = 2 * b - 1
        n[n_boundary] = 2 * b - 1
        # print(m.max())
        # print(m.min())
        # print(n.max())
        # print(n.min())
        # print("happy devoxelizing"+str(torch.cuda.current_device()))

        return m.cuda(), n.cuda()

    def forward(self, x, pointset):  # pylint: disable=W0221
        # x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        # x = x.view(x.size(0), x.size(1), -1).max(-1)[0]  # [batch, feature]
        #
        # x = self.out_layer(self.bn1(x))
        # transfeat = torch.rand(32, 64, 64)  # temp
        # return F.log_softmax(x, dim=1), transfeat

        # layers = []

        # x1 = x[..., 0]
        # x2 = x[..., 1]
        # x3 = x[..., 2]
        #
        # x1 = self.sequential(x1)  # [batch, feature, beta, alpha, gamma]
        # x1 = so3_integrate(x1)  # [batch, feature]
        # layers.append(x1)
        #
        # x2 = self.sequential_2(x2)  # [batch, feature, beta, alpha, gamma]
        # x2 = so3_integrate(x2)  # [batch, feature]
        # layers.append(x2)

        # x3 = self.sequential(x3)  # [batch, feature, beta, alpha, gamma]
        # x3 = so3_integrate(x3)  # [batch, feature]
        # layers.append(x3)

        # combined_feature = torch.cat(layers, dim=1)
        # combined_feature = self.out_layer(combined_feature)
        # return F.log_softmax(combined_feature, dim=1)

        x_original = x.detach().cpu().numpy()
        # print("sequential begin")
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        bandwidth = x.size(2) // 2

        try:
            m, n = self.devoxelize(pointset, bandwidth)
        except Exception:
            print("shit happened during devoxelizing")
        # intregrate all the distributed feature in gamma dimmension
        x = torch.sum(x, dim=-1).squeeze(-1)  # [batch, feature, beta, alpha]
        x_modified = x.detach().cpu().numpy()
        # Compute quadrature weights for the grid
        split = 4
        w = setup_so3_integrate(bandwidth * split, device_type=x.device.type, device_index=x.device.index)  # [beta]
        w = w * bandwidth * 2 * 1000
        # w = w.view(1, 1, 2 * bandwidth, 1)
        # x = x * w # [batch, feature, beta, alpha]

        # devoxelize
        spherical_features = []

        for index in range(x.size(0)):
            try:
                # print("here we begin")
                spherical_features.append(x[index, :, m[index, :], n[index, :]])  # [feature, points]

            except Exception:
                print("shit, we catch it")

        spherical_features = [torch.unsqueeze(array, dim=0) for array in spherical_features]  # [1, feature, points]
        spherical_features = torch.cat(spherical_features, dim=0)  # [batch, feature, points]
        spherical_features = self.bn1(spherical_features)  # [batch, feature, points]
        # spherical_features = torch.max(spherical_features, 2, keepdim=True)[0] #[batch, feature, 1]
        # spherical_features = spherical_features.squeeze() # [batch, feature]

        # spherical_features = self.relu(self.out_layer_1(self.bn1(spherical_features)))
        # spherical_features = self.out_layer(self.bn2(spherical_features))

        # print(m.cpu().size())
        # print(n.cpu().size())
        # print(spherical_features.cpu().size())
        pointset = pointset.transpose(2, 1)  # [batch, xyz, points]
        point_features, transfeat = self.point_net(pointset, spherical_features)  # [batch, feature], [batch, 64, 64]
        # transfeat = torch.rand(32, 64, 64) #temp

        # x = torch.cat([point_features_global, point_features_local + spherical_features], 2)

        # x = torch.max(x, 1, keepdim=True)[0]

        # x = so3_integrate(x)  # [batch, feature]
        #
        # x = x.squeeze()
        # x = self.out_layer_1(x)
        # x = self.out_layer(x)
        return F.log_softmax(point_features, dim=1), transfeat


class Architecture(nn.Module):
    def __init__(self, nclasses):
        super().__init__()

        self.features = [3, 64, 128]
        self.bandwidths = [24, 24]

        self.stn = STN3d(3)

        self.SphericalBlock_1 = SphericalBlock(features=[3, 64],
                                               bandwidths=[24, 16])
        self.PointNetBlock_1 = PointNetBlock(channel=[3, 64], rotated_kernal=True)
        self.SphericalBlock_2 = SO3Block(features=[64, 128],
                                               bandwidths=[16, 16])
        self.PointNetBlock_2 = PointNetBlock(channel=[64, 128])

        #Convolution for fused features
        self.conv1 = torch.nn.Conv1d(128, 1024, 1)
        self.bn = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k=64)

        # Linear
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out_layer = nn.Linear(256, nclasses)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()


        # spherical loss
        self.spherical_loss = spherical_loss(nclasses)

        # adaptive weight
        self.adpative_weight = AdaptiveWeight()

        # Spherical Invariance
        self.rotational_invariance = Rotational_Invariance(dimension=1)
    def forward(self, x, pointset):
        pointset = pointset.transpose(2, 1)  # [batch, xyz, points]
        trans = self.stn(pointset) # [batch, 3, 3]
        # pointset = pointset.transpose(2, 1)
        # pointset = torch.bmm(pointset, trans)
        # pointset = pointset.transpose(2, 1)

        point_feature = self.PointNetBlock_1(pointset)
        spherical_feature, x_so3 = self.SphericalBlock_1(x, pointset) # [batch, feature, points], [batch, feature, beta, alpha, gamma]

        # feature transform
        trans_feat = self.fstn(point_feature)
        # point_feature = point_feature.transpose(2, 1)
        # point_feature = torch.bmm(point_feature, trans_feat)
        # point_feature = point_feature.transpose(2, 1)


        point_feature = self.PointNetBlock_2(point_feature+spherical_feature)
        spherical_feature, x = self.SphericalBlock_2(x_so3, pointset)

        # Convolution for fused features
        # point_py = point_feature.detach().cpu().numpy()
        # spherical_py = spherical_feature.detach().cpu().numpy()
        point_feature = F.relu(self.bn(self.conv1(point_feature + spherical_feature))) # [batch, feature, points]
        # point_feature_relu = point_feature.detach().cpu().numpy()
        # Max-pooling
        point_feature = torch.max(point_feature, 2, keepdim=True)[0] # [batch, feature, 1]
        # Rotational Invariance
        # self.rotational_invariance(point_feature.transpose(2, 1))
        point_feature = point_feature.view(-1, 1024)



        point_feature = F.relu(self.bn1(self.fc1(point_feature)))
        point_feature = F.relu(self.bn2(self.dropout(self.fc2(point_feature))))
        point_feature = self.out_layer(point_feature)

        spherical_global_feature = self.spherical_loss(x)
        # define the weight for different branches
        alpha = self.adpative_weight(trans)
        # alpha = 1
        print(torch.mean(alpha))
        # print(alpha)
        global_feature = alpha * point_feature + (1 - alpha) * spherical_global_feature
        return F.log_softmax(global_feature, dim=1), trans_feat, \
               F.log_softmax(spherical_global_feature), F.log_softmax(point_feature)

class spherical_loss(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.out_layer = nn.Linear(128, nclasses)
        self.bn = nn.BatchNorm1d(40)
    def forward(self, x):
        x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta]
        x = torch.sum(x, dim=-1).squeeze(-1)  # [...]
        x = self.bn(self.out_layer(x))
        # return F.log_softmax(x, dim=1)
        return x

class AdaptiveWeight(nn.Module):
    def __init__(self):
        super(AdaptiveWeight, self).__init__()
        self.pre_layer = nn.Linear(9, 9)
        self.bn = torch.nn.BatchNorm1d(9)
        self.out_layer = nn.Linear(9, 1)
    def forward(self, trans):
        x = self.bn(self.pre_layer(trans.view(-1, 9)))
        return F.sigmoid(self.out_layer(x))


class Rotational_Invariance(nn.Module):
    def __init__(self, dimension=1):
        super(Rotational_Invariance, self).__init__()
        self.conv1 = torch.nn.Conv1d(dimension, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x