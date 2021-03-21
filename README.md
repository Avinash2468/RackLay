# RackLay: Multi-Layer Layout Estimation for Warehouse Racks

[Meher Shashwat Nigam](https://github.com/ShashwatNigam99), [Avinash Prabhu](https://avinash2468.github.io/), Anurag Sahu, Puru Gupta, [Tanvi Karandikar](https://tanvi141.github.io/), N. Sai Shankar, [Ravi Kiran Sarvadevabhatla](https://ravika.github.io), and [K. Madhava Krishna](http://robotics.iiit.ac.in)

#### [Link to paper](https://arxiv.org/abs/2103.09174)
#### [Link to code](https://github.com/Avinash2468/RackLay)

<!-- ####  [Video]( https://youtu.be/1hdl3W-MlXo) -->
<!-- [Paper](https://arxiv.org/abs/2002.08394) -->
<!-- #### Accepted to [WACV 2020](http://wacv20.wacv.net/) -->

<p align="center">
    <img src="assets/teaser.png" />
</p>

## [YouTube Video](https://www.youtube.com/watch?v=1hdl3W-MlXo)

<iframe width="800" src="https://www.youtube.com/embed/1hdl3W-MlXo" align="center" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Abstract

To this end, we present RackLay, a deep neural network for real-time shelf layout estimation from a single image. Unlike previous layout estimation methods which provide a single layout for the dominant ground plane alone, RackLay estimates the top-view \underline{and} front-view layout for each shelf in the considered rack populated with objects. RackLay's architecture and its variants are versatile and estimate accurate layouts for diverse scenes characterized by varying number of visible shelves in an image, large range in shelf occupancy factor and varied background clutter. Given the extreme paucity of datasets in this space and the difficulty involved in acquiring real data from warehouses, we additionally release a flexible synthetic dataset generation pipeline WareSynth which allows users to control the generation process and tailor the dataset according to contingent application. The ablations across architectural variants and comparison with strong prior baselines vindicate the efficacy of RackLay as an apt architecture for the novel problem of multi-layered layout estimation. We also show that fusing the top-view and front-view enables 3D reasoning applications such as metric free space estimation for the considered rack.

<p align="center">
    <img src="assets/double_decoder.png" />
</p>
