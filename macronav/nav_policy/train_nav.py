import os
import threading
import time

import config.train_param as train_param
import torch
from utils.misc import MetricLogger, get_logger, print_params, set_seed
from macronav.nav_policy.models.sac_trainer import SACTrainer
from utils.train_runtime import (
    TRAIN_METRIC_NAMES,
    JobLauncher,
    TrainingRuntimeState,
    create_ray_runner_class,
    init_ray,
    monitor_replay_buffer,
    ray,
)

ray_context = init_ray(train_param)
print(ray_context.dashboard_url)

# Keep Ray resource binding in this file to stay compatible with the original startup pattern.
RayRunner = create_ray_runner_class(train_param.NUM_GPU / train_param.NUM_AGENT)


def main():
    set_seed(train_param.SEED)
    train_param.save_config_artifacts(train_param.CONFIG_DICT, train_param.__file__)
    os.makedirs(train_param.MODEL_PATH, exist_ok=True)
    os.makedirs(train_param.GIF_PATH, exist_ok=True)

    config = {
        "project": train_param.WANDB_PROJECT,
        "entity": train_param.WANDB_ENTITY,
        "name": train_param.EXP_NAME,
        "config": train_param.CONFIG_DICT,
        "log_dir": train_param.TB_PATH,
    }
    log_writer = MetricLogger(use_wandb=train_param.USE_WANDB, config=config)
    runtime_state = TrainingRuntimeState(
        replay_buffer_keys=train_param.REPLAY_BUFFER_KEYS,
        replay_buffer_size=train_param.REPLAY_BUFFER_SIZE,
        metric_names=TRAIN_METRIC_NAMES,
    )

    logger = get_logger(train_param.TB_PATH)
    print_params(logger, train_param)

    device = torch.device("cuda") if train_param.TRAIN_USE_GPU else torch.device("cpu")
    local_device = torch.device("cuda") if train_param.DATA_USE_GPU else torch.device("cpu")
    trainer = SACTrainer(train_param, device, local_device)
    trainer.maybe_load_checkpoint(logger, runtime_state)
    runtime_state.update_weights(trainer.build_worker_weights())

    meta_agents = [RayRunner.remote(i, train_param.CONFIG_DICT) for i in range(train_param.NUM_AGENT)]
    job_launcher = JobLauncher(
        meta_agents,
        state=runtime_state,
        replay_buffer_size=train_param.REPLAY_BUFFER_SIZE,
        logger=logger,
    )
    job_launcher.start()
    replay_buffer_monitor = threading.Thread(
        target=monitor_replay_buffer,
        args=(runtime_state, train_param.MAX_SAMPLE, train_param.EXP_NAME),
        daemon=True,
    )
    replay_buffer_monitor.start()

    # collect data from worker and do training
    try:
        while runtime_state.curr_samples <= train_param.MAX_SAMPLE:
            if not trainer.replay_ready(runtime_state):
                time.sleep(1)
                continue

            trainer.maybe_log_training_start(logger, runtime_state)
            trainer.run_iteration(runtime_state, TRAIN_METRIC_NAMES)
            trainer.maybe_flush_logs(runtime_state, log_writer)
            runtime_state.update_weights(trainer.build_worker_weights())
            trainer.maybe_save_checkpoints(runtime_state)

        print(f"Experiment {train_param.EXP_NAME} finished")
        job_launcher.stop()
        for a in meta_agents:
            ray.kill(a)
        log_writer.close()
        time.sleep(5)
        exit(0)

    except KeyboardInterrupt:
        logger.info("CTRL_C pressed. Killing remote workers")
        job_launcher.stop()
        for a in meta_agents:
            ray.kill(a)
        log_writer.close()
        time.sleep(3)
        exit(0)


if __name__ == "__main__":
    main()
