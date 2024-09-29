import os
import pickle
from .oxford_pets import OxfordPets


# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase
from datasplit import partition_data
from data_utils import prepare_data_office, prepare_data_office_partition_train

# @DATASET_REGISTRY.register()
class Office(DatasetBase):
    dataset_dir = "office_caltech_10"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_fed")
        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        ori_train, ori_test, classnames, lab2cname = prepare_data_office(cfg, root)

        total_train = []
        for domain in ori_train:
            for datum in domain:
                total_train.append(datum)
        test = []
        for domain in ori_test:
            for datum in domain:
                test.append(datum)

        p_val = 2 / 7  # 0.5 train, 0.2 val, 0.3 test
        train, val = OxfordPets.split_trainval(total_train, p_val=p_val)

        if num_shots >= 1:
            seed = cfg.SEED
            if cfg.TRAINER.NAME == "Baseline":
                preprocessed = os.path.join(self.baseline_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                # with open(preprocessed, "wb") as file:
                #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        if not cfg.DATASET.IID and cfg.DATASET.BETA != -1:
            output_dset = self.generate_dirichlet_federated_dataset(total_train, test, num_shots=num_shots,
                                                                    num_users=cfg.DATASET.USERS, beta=cfg.DATASET.BETA, is_iid=cfg.DATASET.IID,
                                                                    repeat_rate=cfg.DATASET.REPEATRATE)
            federated_train_x, federated_test_x = output_dset[0], output_dset[1]
        elif cfg.DATASET.USERS > 0 and cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(total_train, num_shots=num_shots,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                               num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                               repeat_rate=cfg.DATASET.REPEATRATE)
            print("federated all dataset")
        elif cfg.DATASET.USERS > 0 and not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(total_train, num_shots=num_shots,
                                                                        num_users=cfg.DATASET.USERS,
                                                                        is_iid=cfg.DATASET.IID,
                                                                        repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                               num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                               repeat_rate=cfg.DATASET.REPEATRATE)
            print("fewshot federated dataset")
        else:
            federated_train_x = None


        super().__init__(train_x=train, federated_train_x=federated_train_x, val=val,
                         federated_test_x=federated_test_x,
                         test=test)




