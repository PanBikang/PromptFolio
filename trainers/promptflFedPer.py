import copy
import torch

# @TRAINER_REGISTRY.register("PromptFL")
from trainers.promptfl import PromptFL
class PromptFLFedPer(PromptFL):

    def fed_aggregate_model(self, idx_users):
        global_list = []
        self.per_length = self.local_info[0]["prompt_learner.ctx"].shape[0] // 2  # [8, 512] -> 4
        for idx in idx_users:
            global_list.append(self.local_info[idx]["prompt_learner.ctx"][:self.per_length])
        self.global_info = copy.deepcopy(self.average_weights(global_list, idx_users, islist=True))

    def fed_download_model(self, idx):
        self.model.load_state_dict(self.local_info[idx])
        if self.global_info is not None:
            prompt = torch.cat([self.global_info, self.local_info[idx]['prompt_learner.ctx'][self.per_length:]], dim=0)
            self.model.load_state_dict({'prompt_learner.ctx': prompt}, strict=False)



