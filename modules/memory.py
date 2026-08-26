import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

    def __init__(
        self,
        n_nodes:int,
        memory_dimension:int,
        input_dimension,
        message_dimension=None,
        device="cpu",
        combination_method="sum",
    ):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device

        self.combination_method = combination_method

        self.register_buffer('memory', torch.zeros((self.n_nodes, self.memory_dimension)))
        self.register_buffer('last_update', torch.zeros(self.n_nodes))

        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model
        # Zero out existing buffers in-place instead of creating new ones
        self.memory.zero_()
        self.last_update.zero_()

        # if a key is not available it will return empty list
        self.messages = defaultdict(list)

        # if a key is not available it will return empty list
        self.messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        # Use standard .clone() without .data
        return self.memory.clone(), self.last_update.clone(), messages_clone

    def restore_memory(self, memory_backup):
        # Safely overwrite the buffer values in-place
        self.memory.copy_(memory_backup[0])
        self.last_update.copy_(memory_backup[1])

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_()

        # Detach all stored messages
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []
