import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class EdgeConvWithEdgeAttr(MessagePassing):
    def __init__(self, mlp):
        super().__init__(aggr='max')  # EdgeConv 
        self.mlp = mlp

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim], edge_attr: [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate x_i (central), x_j (neighbor), and edge_attr
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(msg_input)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
class InteractiveGNN(nn.Module):
    def __init__(self, node_dim=TODO, edge_dim=TODO, hidden_dim=TODO, out_dim=TODO):
        super().__init__()
        self.edge_mlp1 = MLP(node_dim * 2 + edge_dim, [64], hidden_dim)
        self.edge_mlp2 = MLP(hidden_dim * 2 + edge_dim, [64], hidden_dim)

        self.conv1 = EdgeConvWithEdgeAttr(self.edge_mlp1)
        self.conv2 = EdgeConvWithEdgeAttr(self.edge_mlp2)

        self.readout = MLP(hidden_dim, [64], out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # print(f"InteractiveGNN input shapes: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}, batch={batch.shape}")
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        x_pool = global_mean_pool(x, batch)
        z = self.readout(x_pool)
        return z

    
    def build_interaction_graph(self, obs_seq, obs_horizon):
        """
        this function should provide the node features and edge features for the GNN based on the observation sequence.
        notice that each return values should prepare # obs_horizon, each return values per obs_horizon
        obs_seq: tensor of shape [B, obs_horizon, obs_dim]
        node_features: [B, obs_horizon, node_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        B, obs_horizon, obs_dim = obs_seq.shape
        # assume that obs_horizon is just one, later we can extend to multiple timesteps by stacking the graph features across time
        obs_now = obs_seq[:, 0, :]  # [B, obs_dim]
        device = obs_seq.device

        # node features:
        # obstacle, box, vehicle have their (x, y) positions respectively
        obstacle_feat = obs_now[:, :2]  # [B, 2]
        box_feat = obs_now[:, 2:4]    # [B, 2]
        vehicle_feat = obs_now[:, 4:6] # [B, 2]

        # all nodes have the same dimension so we just skip zero-padding for now
        all_nodes = torch.cat([
            obstacle_feat.unsqueeze(1),  # [B, 1, 2]
            box_feat.unsqueeze(1),       # [B, 1, 2]
            vehicle_feat.unsqueeze(1)    # [B, 1, 2]
        ], dim=1)  # [B, num_nodes=3, node_dim=2]

        type_tensor = torch.tensor([
            [1, 0, 0],  # obstacle
            [0, 1, 0],  # box
            [0, 0, 1]   # vehicle
        ], device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, num_nodes=3, type_dim=3]
        node_features = torch.cat([all_nodes, type_tensor], dim=-1)  # [B, num_nodes=3, node_dim=5]
        node_features = node_features.view(B * 3, -1)  # [B*num_nodes, node_dim]

        # pose_table:
        obstacle_xy = obstacle_feat.unsqueeze(1)  # [B, 1, 2]
        box_xy = box_feat.unsqueeze(1)            # [B, 1, 2]
        vehicle_xy = vehicle_feat.unsqueeze(1)    # [B, 1, 2]
        pose_table = torch.cat([obstacle_xy, box_xy, vehicle_xy], dim=1)  # [B, num_nodes=3, 2]
        pose_table = pose_table.view(B * 3, 2)  # [B*num_nodes, 2]

        # the batch vector for global pooling:
        # batch = [0, 0, 0, 1, 1, 1, ..., B-1, B-1, B-1] 
        # where each episode's nodes are grouped together
        batch = torch.arange(B, device=device).unsqueeze(1).repeat(1, 3).view(-1)  # [B*num_nodes]

        # construct edge features

        local_edges = [(0, 1), (0, 2)]  # vehicle -> box, obstacle
        local_edges += [(1, 2)]  # box -> obstacle
        local_edges += [(dst, src) for (src, dst) in local_edges]  # add reverse edges

        num_edges_per_graph = len(local_edges)
        local_edges_tensor = torch.tensor(local_edges, dtype=torch.long,device=device) # [E, 2]
        src_local, dst_local = local_edges_tensor[:, 0], local_edges_tensor[:, 1]  # [E]

        # handle batch of graphs by offsetting
        batch_offset = torch.arange(B, device=device) * 3  # [B]
        # batch_offset:
        #  [[0,   0,   0,   0,   0,   0  ],
        #  [3,   3,   3,   3,   3,   3  ],
        #  [6,   6,   6,   6,   6,   6  ],
        #  ...
        #  [(B-1)*3, ..., (B-1)*3       ]]   shape: [B, E]
        batch_offset = batch_offset.view(-1, 1).expand(-1, num_edges_per_graph)  # [B, E]

        src_all = src_local.view(1, -1) + batch_offset  # [B, E]
        dst_all = dst_local.view(1, -1) + batch_offset  # [B, E]

        edge_index = torch.stack([
            src_all.reshape(-1),  # [B*E]
            dst_all.reshape(-1)   # [B*E]
        ], dim=0)  # [2, B*E]

        # edge features: we use relative positions
        src, dst = edge_index
        src_pos = pose_table[src]  # [B*E, 2]
        dst_pos = pose_table[dst]  # [B*E, 2]
        edge_attr = dst_pos - src_pos  # [B*E, 2]

        return node_features, edge_index, edge_attr, batch


