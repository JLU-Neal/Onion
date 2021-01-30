import torch.nn as nn
import torch.nn.functional as F
import torch
from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate, setup_so3_integrate
import math

class SphericalBlock(nn.Module):
    def __init__(self, features=None, bandwidths=None):
        super().__init__()

        if bandwidths is None:
            bandwidths = [24, 24]
        if features is None:
            features = [3, 100]
        self.features = features
        self.bandwidths = bandwidths

        # define the number of layers.
        self.num_layers = 1

        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        # =====================If we could chose another gridding method=====================
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2Convolution(self.features[0], int(self.features[1] / self.num_layers), self.bandwidths[0],
                                      self.bandwidths[1], grid))
        sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

        self.bn1 = nn.BatchNorm1d(self.features[-1])


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
        #need to mind the boundary issues
        m_boundary = m >= 2*b
        n_boundary = n >= 2*b
        m[m_boundary] = 2*b - 1
        n[n_boundary] = 2*b - 1
        # print(m.max())
        # print(m.min())
        # print(n.max())
        # print(n.min())
        # print("happy devoxelizing"+str(torch.cuda.current_device()))

        return m.cuda(), n.cuda()

    def forward(self, x, pointset):  # pylint: disable=W0221
        pointset = pointset.transpose(2, 1)

        # x_original = x.detach().cpu().numpy()
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        bandwidth = x.size(2) // 2

        try:
            m, n = self.devoxelize(pointset, bandwidth)
        except Exception:
            print("shit happened during devoxelizing")
        # intregrate all the distributed feature in gamma dimension
        x_so3 = x
        x = torch.sum(x, dim=-1).squeeze(-1)  #[batch, feature, beta, alpha]
        # x_modified = x.detach().cpu().numpy()
        # Compute quadrature weights for the grid, only necessary for last layer
        # split = 1
        # w = setup_so3_integrate(bandwidth * split, device_type=x.device.type, device_index=x.device.index)  # [beta]
        # w = w *bandwidth
        # w = w.view(1, 1, 2 * bandwidth, 1)
        # x = x * w # [batch, feature, beta, alpha]
        # x_weight = x.detach().cpu().numpy()

        # devoxelize
        spherical_features = []


        for index in range(x.size(0)):
            try:
                spherical_features.append(x[index, :, m[index, :], n[index, :]]) # [feature, points]

            except Exception:
                print("shit, we catch it")

        spherical_features = [torch.unsqueeze(array, dim=0) for array in spherical_features] # [1, feature, points]
        spherical_features = torch.cat(spherical_features, dim=0) # [batch, feature, points]
        # spherical_np = spherical_features.detach().cpu().numpy()
        spherical_features = self.bn1(spherical_features) # [batch, feature, points]
        # spherical_bn = spherical_features.detach().cpu().numpy()
        return F.relu(spherical_features), x_so3


class PointNetBlock(nn.Module):
    def __init__(self, channel=None, rotated_kernal=False):
        super(PointNetBlock, self).__init__()
        if channel is None:
            channel = [3, 64]
        self.conv1 = torch.nn.Conv1d(channel[0], channel[1], 1)
        self.bn1 = nn.BatchNorm1d(channel[1])
        # To construct rotation matrix for different angle, to rotate the kernal
        # [rotate_times, 3, 3]
        self.rotated_kernal = rotated_kernal
        if self.rotated_kernal:
            angle = math.pi/6
            rotmat_list = []
            self.batchnorma_list = []
            for time in range (int(2*math.pi/angle-1)):
                cur_angle = (time + 1) * angle
                rotmat_list.append(torch.tensor([[math.cos(cur_angle), -math.sin(cur_angle), 0],
                                                 [math.sin(cur_angle), math.cos(cur_angle), 0],
                                                 [0, 0, 1]]).unsqueeze(dim=0))
                # self.batchnorma_list.append(nn.BatchNorm1d(channel[1]))
            # for index in range(len(self.batchnorma_list)):
            #     self.batchnorma_list[index] = self.batchnorma_list[index].cuda()
            self.rotmat_list = torch.cat(rotmat_list, dim=0).cuda()

    def forward(self, x):
        x_original = x
        feature_list = []
        feature_list.append(self.conv1(x).unsqueeze(3))
        if self.rotated_kernal:
            weight = self.conv1.weight
            weight_list = []

            # Convolution of kernals with different angles
            for index in range(self.rotmat_list.shape[0]):
                cur_weight = torch.mm(weight.squeeze(), self.rotmat_list[index].cuda())
                cur_weight = cur_weight.unsqueeze(dim=2).cuda()
                weight_list.append(cur_weight)

            for index in range(len(weight_list)):
                # self.batchnorma_list[index] = self.batchnorma_list[index].cuda()
                feature_list.append(F.conv1d(x_original, weight_list[index]).unsqueeze(3))
        feature_list = torch.cat(feature_list, dim=3)
        x = torch.mean(feature_list, dim=3)
        x = F.relu(self.bn1(x))
        return x



class SO3Block(nn.Module):
    def __init__(self, features=None, bandwidths=None):
        super().__init__()

        if bandwidths is None:
            bandwidths = [24, 24]
        if features is None:
            features = [3, 100]
        self.features = features
        self.bandwidths = bandwidths

        # define the number of layers.
        self.num_layers = 1

        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        # =====================If we could chose another gridding method=====================
        grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * bandwidths[0], n_beta=1, n_gamma=1)
        sequence.append(SO3Convolution(features[0], features[1], bandwidths[0], bandwidths[1], grid))
        sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

        self.bn1 = nn.BatchNorm1d(self.features[-1])


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
        #need to mind the boundary issues
        m_boundary = m >= 2*b
        n_boundary = n >= 2*b
        m[m_boundary] = 2*b - 1
        n[n_boundary] = 2*b - 1
        # print(m.max())
        # print(m.min())
        # print(n.max())
        # print(n.min())
        # print("happy devoxelizing"+str(torch.cuda.current_device()))

        return m.cuda(), n.cuda()

    def forward(self, x, pointset):  # pylint: disable=W0221
        pointset = pointset.transpose(2, 1)

        # x_original = x.detach().cpu().numpy()
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        bandwidth = x.size(2) // 2

        try:
            m, n = self.devoxelize(pointset, bandwidth)
        except Exception:
            print("shit happened during devoxelizing")
        # intregrate all the distributed feature in gamma dimmension
        x = torch.sum(x, dim=-1).squeeze(-1)  #[batch, feature, beta, alpha]
        # x_modified = x.detach().cpu().numpy()
        # Compute quadrature weights for the grid
        split = 1
        w = setup_so3_integrate(bandwidth * split, device_type=x.device.type, device_index=x.device.index)  # [beta]
        w = w *bandwidth
        w = w.view(1, 1, 2 * bandwidth, 1)
        x = x * w # [batch, feature, beta, alpha]
        # x_weight = x.detach().cpu().numpy()

        # devoxelize
        spherical_features = []


        for index in range(x.size(0)):
            try:
                spherical_features.append(x[index, :, m[index, :], n[index, :]]) # [feature, points]

            except Exception:
                print("shit, we catch it")

        spherical_features = [torch.unsqueeze(array, dim=0) for array in spherical_features] # [1, feature, points]
        spherical_features = torch.cat(spherical_features, dim=0) # [batch, feature, points]
        # spherical_np = spherical_features.detach().cpu().numpy()
        spherical_features = self.bn1(spherical_features) # [batch, feature, points]
        # spherical_bn = spherical_features.detach().cpu().numpy()
        return F.relu(spherical_features), x