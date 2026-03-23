from .metric import CmapvocPool
import numpy as np
import json


class Log:
    def __init__(self, rank, args, description):
        self.args = args
        self.rank = rank
        self.description = description

        self.logit = []
        self.label = []
        self.loss = []

    def log(self, logit, label, loss):
        logit = logit.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()

        loss = [loss.detach().item(), ]

        self.logit += logit
        self.label += label
        self.loss += loss

    def write(self):
        logit = np.asarray(self.logit, dtype=np.float32)
        label = np.asarray(self.label, dtype=np.float32)
        loss = np.asarray(self.loss, dtype=np.float32)
        np.savez(f"./{self.args.checkpoint_dir}/temp/{self.rank}_{self.description}.npz", logit=logit, label=label, loss=loss)


def analyse_mAP(args):
    world_size = args.world_size
    train_data = [np.load(f"./{args.checkpoint_dir}/temp/{i}_training.npz") for i in range(world_size)]
    eval_data = [np.load(f"./{args.checkpoint_dir}/temp/{i}_validating.npz") for i in range(world_size)]
    ema_data = [np.load(f"./{args.checkpoint_dir}/temp/{i}_ema_validating.npz") for i in range(world_size)]

    train_logit = np.concatenate([train_data[i]["logit"] for i in range(world_size)], axis=0)
    train_label = np.concatenate([train_data[i]["label"] for i in range(world_size)], axis=0)
    train_loss = np.concatenate([train_data[i]["loss"] for i in range(world_size)], axis=0)

    eval_logit = np.concatenate([eval_data[i]["logit"] for i in range(world_size)], axis=0)
    eval_label = np.concatenate([eval_data[i]["label"] for i in range(world_size)], axis=0)
    eval_loss = np.concatenate([eval_data[i]["loss"] for i in range(world_size)], axis=0)

    ema_logit = np.concatenate([ema_data[i]["logit"] for i in range(world_size)], axis=0)
    ema_label = np.concatenate([ema_data[i]["label"] for i in range(world_size)], axis=0)
    ema_loss = np.concatenate([ema_data[i]["loss"] for i in range(world_size)], axis=0)

    cpor = CmapvocPool()
    cpor.put_job(train_logit, train_label)
    cpor.put_job(eval_logit, eval_label)
    cpor.put_job(ema_logit, ema_label)

    aps = cpor.get_result()

    train_aps, eval_aps, ema_aps = np.split(aps, 3, axis=0)
    train_map, eval_map, ema_map = float(train_aps.mean()), float(eval_aps.mean()), float(ema_aps.mean())

    train_loss = float(np.mean(train_loss))
    eval_loss = float(np.mean(eval_loss))
    ema_loss = float(np.mean(ema_loss))

    train_aps, eval_aps, ema_aps = train_aps.tolist(), eval_aps.tolist(), ema_aps.tolist()

    return ((train_aps, train_map, train_loss),
            (eval_aps, eval_map, eval_loss),
            (ema_aps, ema_map, ema_loss))


class FileLog:
    def __init__(self, args):
        self.fp = open(f"./{args.checkpoint_dir}/log.txt", "a")

    def log(self, epoch, train_msg, eval_msg, ema_msg):
        train_aps, train_map, train_loss = train_msg
        eval_aps, eval_map, eval_loss = eval_msg
        ema_aps, ema_map, ema_loss = ema_msg
        temp = {
            "epoch": epoch,
            "train_aps": train_aps,
            "train_map": train_map,
            "train_loss": train_loss,
            "eval_aps": eval_aps,
            "eval_map": eval_map,
            "eval_loss": eval_loss,
            "ema_aps": ema_aps,
            "ema_map": ema_map,
            "ema_loss": ema_loss,
        }
        temp = json.dumps(temp)
        self.fp.write(temp + "\n")
        self.fp.flush()
        temp = {
            "epoch": epoch,
            "train_map": train_map,
            "train_loss": train_loss,
            "eval_map": eval_map,
            "eval_loss": eval_loss,
            "ema_map": ema_map,
            "ema_loss": ema_loss,
        }
        temp = json.dumps(temp)
        print(temp)

    def close(self):
        self.fp.close()