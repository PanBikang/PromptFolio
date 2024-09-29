import os
import pickle
import random

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


# @DATASET_REGISTRY.register()
class DescribableTextures(DatasetBase):
    dataset_dir = "dtd"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_fed")
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            total_train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            total_train, val, test = self.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        if num_shots >= 1:
            seed = cfg.SEED
            if cfg.TRAINER.NAME == "Baseline":
                preprocessed = os.path.join(self.baseline_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(total_train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        if not cfg.DATASET.IID and cfg.DATASET.BETA != -1:
            output_dset = self.generate_dirichlet_federated_dataset(total_train, test, num_shots=num_shots,
                                                                    num_users=cfg.DATASET.USERS, beta=cfg.DATASET.BETA, is_iid=cfg.DATASET.IID,
                                                                    repeat_rate=cfg.DATASET.REPEATRATE)
            federated_train_x, federated_test_x = output_dset[0], output_dset[1]
        elif cfg.DATASET.USERS > 0 and cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(total_train, num_shots=num_shots,
                                                                        num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            print("federated all dataset")
        elif cfg.DATASET.USERS > 0 and not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(total_train, num_shots=num_shots,num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            print("fewshot federated dataset")
        else:
            federated_train_x = None

        super().__init__(train_x=train, federated_train_x=federated_train_x, val=val, federated_test_x=federated_test_x, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============

        # convert all the class under image dir to a list
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test
