<img align="left" width="100" src="logo.png" alt="DAC4AutoML Competition Logo">

# DAC4AutoML Competition 2022

Welcome to our DAC4AutoML competition! We challenge you to dynamically adapt hyperparameters for Deep Learning (DL) and Reinforcement Learning (RL). For this we provide two challenges: DAC4SGD and DAC4RL. For details on the challenges as well as the general competition rules, check out the CodaLab pages below.

### Motivation

Hyperparameters in Deep Learning (DL) and Reinforcement Learning (RL) are often adjusted online, while learning. This dynamic adaptation is most commonly achieved using handcrafted heuristics (Battiti et al., 2008; Kingma and Ba, 2015; Loshchilov and Hutter, 2017; Drake et al., 2020) and, more rarely, using a (meta-)learned control policy (Daniel et al., 2016; Sharma et al., 2019; Gomoluch et al., 2020; Almeida et al., 2021). We believe there is still a lot of potential for finding more such sophisticated solutions that generalize better to different problem settings. Therefore, this competition builds upon the Dynamic Algorithm Configuration (DAC) framework (Biedenkapp et al., 2020) to provide a competition setting that tests both the quality of hyperparameter configuration policies as well as their generalization capabilities.

#### Dynamic Algorithm Configuration

Dynamic Algorithm Configuration is a generalization of the well-known paradigms of Algorithm Configuration and Per-Instance Algorithm Configuration (Birattari et al., 2002; Ansótegui et al., 2009; Kadioglu et al., 2010; Xu et al., 2010; Lindauer et al., 2022). Instead of finding a single configuration for a set of problem instances or a configuration per instance, DAC finds a policy for adjusting the configuration at every step of the target algorithm (see Figure down below). This leads to a configuration that is able to adapt to the current state of algorithm execution. The effectiveness of this approach has been demonstrated far before DAC was formally introduced, with early successful applications in, e.g., recursive algorithm selection (Lagoudakis and Littman, 2000), heuristic optimization (Battiti and Campigotto, 2012; López-Ibánez and Stützle, 2014; Kadioglu et al., 2017; Sae-Dan et al., 2020) or limited settings in machine learning (Daniel et al., 2016; Hansen, 2016; Fu, 2016; Xu et al., 2017, 2019; Almeida et al., 2021). The DAC framework presents a novel view on Algorithm Configuration, unifying previous research on dynamic hyperparameter adaptations from different fields and formalizing its objectives (Biedenkapp et al., 2020). Since then, there has been steady progress with significant improve- ments in domains like AI Planning (Speck et al., 2021) and Evolutionary Computation (Shala et al., 2020). We hope to spark similar progress in DL and RL using the framework of DAC in this competition. Therefore, the competition focuses on discovering dynamic hyperparameter schedules for two problem settings, using any method you can come up with.

<img align="center" width="80%" src="dacloop.png" alt="DAC Loop">

### Challenge: DAC4SGD

While DAC has been applied to Computer Vision problems before (Daniel et al., 2016; Xu et al., 2017; Almeida et al., 2021), the research thus far has yet to yield practical online hyperparameter adaptation policies. Furthermore, the associated code often has not even been released and the experimental setups have been hard to replicate from the papers alone. In short, the challenge can be described as follows:

- Goal: Dynamically adapt the learning rate of the SGD optimizer
- Based on the extended version of the SGDBenchmark included in DACBench.
- Provided baselines: Static learning rate, cosine annealing (Loshchilov and Hutter, 2017), reduce learning rate on pleateu (Pytorch; Paszke et al., 2019) and a basic RL agent.
- CodaLab competition page: TBA

### Challenge: DAC4RL

In Reinforcement Learning, dynamic configuration is common but does not currently target transfer or generalization (Jaderberg et al., 2017; Parker-Holder et al., 2020; Awad et al., 2021) either across variations of the same environment or across different environments. As a first step towards tackling this challenge, we will provide variations of five environments through different contexts for the CARL environments (Benjamins et al., 2021). Different contexts would then represent different train and test settings on which the participants’ approaches would be tested. This would explicitly encourage better transfer of hyperparameters between different contexts as envisioned in Kirk et al. (2021) and further progress on the DAC for RL state-of-the-art. In short, the challenge can be described as follows:

- Goal: Dynamically adapt the hyperparameter configuration of a stable_baselines3 agent
- Based on 5 CARL environments
- Provided baselines: Static configurations and learned configuration schedule found by state-of-the-art AutoRL tool PB2 (Parker-Holder et al., 2020)
- CodaLab competition page: TBA

### Awards

We will provide you with certificates of participation (physical as well as digital), including your placement and medals for the top 3 teams of each track. Further prizes for the top 3 teams are copies of the AutoML book signed by the editors. These teams will also receive monetary prizes sponsored by ChaLearn:

- 250$ for first place
- 150$ for second
- 100$ for third

### Organizers
The DAC4AutoML competition is organized by the [AutoML Freiburg-Hannover group](automl.org).


<table>
  <tr>
    <td><a href= "https://www.tnt.uni-hannover.de/staff/eimer/">Theresa Eimer </a> </td>
     <td><a href= "https://www.tnt.uni-hannover.de/staff/eimer/">Raghu Rajan </a> </td>
     <td><a href= "https://www.tnt.uni-hannover.de/staff/mohan/">Aditya Mohan </a> </td>
  </tr>
  <tr>
    <td><img src="theresa.jpg" width="150" height="200"></td>
    <td><img src="raghu.jpg" width="175" height="200"></td>
    <td><img src="aditya.jpg" width="150" height="200"></td>
  </tr>
 </table>
 
<figure class="image">
    <img src="theresa.jpg" width="150" height="200">
    <font size="2">
    <figcaption> <a href= "https://www.tnt.uni-hannover.de/staff/eimer/">Theresa Eimer </a> 
    </figcaption>
    </font>
</figure>
<figure>
    <img src="raghu.jpg" width="175" height="200">
    <font size="2">
    <figcaption> <a href= "https://www.tnt.uni-hannover.de/staff/eimer/">Raghu Rajan </a> 
    </figcaption>
    </font>
</figure>
    <figure>
    <img src="aditya.jpg" width="150" height="200">
    <font size="2">
    <figcaption> <a href= "https://www.tnt.uni-hannover.de/staff/mohan/">Aditya Mohan </a> 
    </figcaption>
    </font>
</figure>

<figure>
    <img src="goktug.jpeg" width="150" height="200">
    <figcaption class="figure-caption text-center"> <a href= "https://www.tnt.uni-hannover.de/staff/eimer/">Göktuğ Karakaşlı</a> 
    </figcaption>
</figure>

<figure>
    <img src="carolin.jpg" width="150" height="200">
    <figcaption> <a href= "https://www.tnt.uni-hannover.de/staff/benjamin/">Carolin Benjamins </a> 
    </figcaption>
</figure>
    
<figure>
    <img src="steven.jpg" width="150" height="200">
    <font size="2">
    <figcaption class="figure-caption text-center"> <a href= "https://www.tnt.uni-hannover.de/staff/mohan/">Steven Adriaensen </a> 
    </figcaption>
    </font>
</figure>

<figure>
    <img src="frank.jpg" width="180" height="200">
    <font size="2">
    <figcaption> <a href= "https://www.tnt.uni-hannover.de/staff/eimer/">Frank Hutter </a> 
    </figcaption>
    </font>
</figure>
<figure>
    <img src="marius.jpg" width="150" height="200">
    <font size="2">
    <figcaption> <a href= "https://www.tnt.uni-hannover.de/staff/lindauer/">Marius Lindauer </a> 
    </figcaption>
    </font>
</figure>
