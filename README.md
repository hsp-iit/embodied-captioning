# Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions

We present a self-supervised method to improve an agent's abilities in describing arbitrary objects while actively exploring a generic environment. This is a challenging problem, as current models struggle to obtain coherent image captions due to different camera viewpoints and clutter. We propose a three-phase framework to fine-tune existing captioning models that enhances caption accuracy and consistency across views via a consensus mechanism. First, an agent explores the environment, collecting noisy image-caption pairs. Then, a consistent pseudo-caption for each object instance is distilled via consensus using a large language model. Finally, these pseudo-captions are used to fine-tune an off-the-shelf captioning model, with the addition of contrastive learning. We analyse the performance of the combination of captioning models, exploration policies, pseudo-labeling methods, and fine-tuning strategies, on our manually labeled test set. Results show that a policy can be trained to mine samples with higher disagreement compared to classical baselines. Our pseudo-captioning method, in combination with all policies, has a higher semantic similarity compared to other existing methods, and fine-tuning improves caption accuracy and consistency by a significant margin.

[[arXiv](https://doi.org/10.48550/arXiv.2504.08531)] [[webpage](https://hsp-iit.github.io/embodied-captioning/)]

<p align="center">
<img src="docs/assets/framework.png" width="920" />
</p>

## Table of Contents
1. [Setup](#setup)
2. [Data](#data)
3. [Demo](#demo)
4. [Contributing](#contributing)
5. [Credits](#credits)
6. [Enquiries, Question and Comments](#enquiries-question-and-comments)
7. [License](#license)


## Setup <a name="setup"></a>
- Clone repository: `git clone https://github.com/hsp-iit/embodied-captioning.git`
- Clone submodules: `git submodule update --init --recursive`
- Create and activate conda env: 
```
conda env create -f env.yml
conda activate SImCa
```
- Install habitat:
```
conda install habitat-sim withbullet -c conda-forge -c aihabitat
cd third_parties/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd -
```

- Install detectron2:
```
cd third_parties
python -m pip install -e detectron2
# Rename folder datasets due to conflicts with transformers.datasets
mv detectron2/datasets detectron2/datasets_det 
cd -
```

- Install pyntcloud:
```
cd third_parties/pyntcloud
python setup.py develop
cd -
```
- Install other dependencies: `pip install -r requirements.txt`
- Install module: `python setup.py develop`
- Modify `base_dir` inside `confs/train_policy.yaml` as the absolute path of this project

   
## Data <a name="data"></a>
All the data are contained inside the `data` directory.

### Example: habitat test scenes
1. Download test-scenes `http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip`
2. Unzip inside project directory

- Suggestion: you can keep data separate and use soft links (`ln -s
  /path/to/dataset /path/to/project/data`)

### Scenes datasets
| Scenes models              | Extract path                                   | Archive size |
| ---                           | ---                                            | ---          |
| [Gibson](#Gibson)          | `data/scene_datasets/gibson/{scene}.glb`       | 1.5 GB       |
| [HM3D](#HM3D) | `data/scene_datasets/hm3d/{split}/00\d\d\d-{scene}/{scene}.basis.glb` | 130 GB        |

#### MatterPort
- Follow instruction in the main [Habitat-lab](https://github.com/facebookresearch/habitat-lab) repository 

#### Gibson
- Fill the form to get Gibson download links at `https://stanfordvl.github.io/iGibson/dataset.html`
- Download Gibson compatible with Habitat (train_val folder) containing the .glb files
- Download gibson tiny with `wget https://storage.googleapis.com/gibson_scenes/gibson_tiny.tar.gz`
- Follow instructions at [Habitat-sim](https://github.com/facebookresearch/habitat-sim) to generate gibson semantic
- Place gibson semantic .ids and .scn files with the .glb and .navmesh files in the folder *data/scene_datasets/gibson*  

### Tasks 
#### Gibson object navigation (used for the main results of the paper)
You can download the task at the following link {ADD LINK}, unzip and put it in `data/datasets/objectnav/gibson/v1.1`

#### Habitat's pointnav
| Task | Scenes | Link | Extract path | Config to use | Archive size |
| --- | --- | --- | --- | --- | --- |
| [Point goal navigation](https://arxiv.org/abs/1807.06757) | Gibson | [pointnav_gibson_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) | `data/datasets/pointnav/gibson/v1/` |  [`datasets/pointnav/gibson.yaml`](configs/datasets/pointnav/gibson.yaml) | 385 MB |
| [Point goal navigation corresponding to Sim2LoCoBot experiment configuration](https://arxiv.org/abs/1912.06321) | Gibson | [pointnav_gibson_v2.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip) | `data/datasets/pointnav/gibson/v2/` |  [`datasets/pointnav/gibson_v2.yaml`](configs/datasets/pointnav/gibson_v2.yaml) | 274 MB |
| [Point goal navigation](https://arxiv.org/abs/1807.06757) | MatterPort3D | [pointnav_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip) | `data/datasets/pointnav/mp3d/v1/` | [`datasets/pointnav/mp3d.yaml`](configs/datasets/pointnav/mp3d.yaml) | 400 MB |
  
## Demo

### Launch an experiment
`python scripts/run_exp.py` run a policy in `train` or `generate` mode. More information below.

### Replay experiment
To replay an experiment, use the following
`python scripts/visualize_exp.py replay.episode_id={ID episode} replay.exp_name={PATH TO EPISODE} replay.modalities="['rgb', 'depth','semantic']"`

### Baselines and environments

The following RL baselines are implemented:
- `goalexplorationbaseline-v0` State: disagreement_t, map_t, agent pose

The following classical baselines are implemented:
- `randomgoalsbaseline`
- `frontierbaseline-v1` (`frontierbaseline-v0` is deprecated)
- `bouncebaseline`
- `rotatebaseline`
- `randombaseline`
- `observeobjectdiscreteactionsbaseline`

The following environments are also implemented:
- `SemanticCuriosity-v0` reward: sum(disagreement_t)
- `SemanticDisagreement-kl` reward: sum(disagreement_t)

### Train goalexploration policy
Start from `confs/habitat/gibson_goal_exploration.yaml`

- `CHECKPOINT_FOLDER` folder in which checkpoints are saved
- `TOTAL_NUM_STEPS` max number of training steps
- under `ppo`:
  - `replanning_steps` how often to run the policy
  - `num_global_steps` how often to train the policy
  - `save_periodic` how often to save a checkpoint
  - `load_checkpoint_path` full path to a checkpoint to load at start
  - `load_checkpoint` set True to load `load_checkpoint_path`
  - `visualize` if True, debug images are shown
  - `mode` set to "train"
- captioner:
  - `gpu_ids` identifiers of the gpus to use
  - `arch_name` name of the architecture to load
  - `model_name`name of the model to load
  - `checkpoint_name` name of the checkpoint to load
  - `debug` print the predicted captions

### Generate from goalexploration policy
Start from `confs/habitat/gibson_goal_exploration.yaml`
- `mode` set to "generate"


## Contributing <a name="contributing"></a>

If you find an error, if you want to suggest a new feature or a change, you can use the issues tab to raise an issue with the appropriate label. 


## Credits <a name="credits"></a>


```
@article{galliena2025embodied,
        title={Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions},
        author={Galliena, Tommaso and Apicella, Tommaso and Rosa, Stefano and Morerio, Pietro and Del Bue, Alessio and Natale, Lorenzo},
        journal={arXiv preprint arXiv:2504.08531},
        year={2025}
}
```


## Enquiries, Question and Comments <a name="enquiries-question-and-comments"></a>

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, please use the Github issue tracker. 


## Licence <a name="license"></a>
This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
