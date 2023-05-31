# HGTN
**abstract**
Graph neural networks (GNNs) have been widely used for graph structure learning and achieved excellent performance in tasks such as node classification and link prediction. Real-world graph networks imply complex and various semantic information and are often referred to as heterogeneous information networks (HINs). Previous GNNs have laboriously modeled heterogeneous graph networks with pairwise relations, in which the semantic information representation for learning is incomplete and severely hinders node embedded learning. Therefore, the conventional graph structure cannot satisfy the demand for information discovery in HINs. In this paper, we propose an end-to-end hypergraph transformer neural network (HGTN) that exploits the communication abilities between different types of nodes and hyperedges to learn higher-order relations and discover semantic information. Specifically, attention mechanisms weigh the importance of semantic information hidden in original HINs to generate useful meta-paths. Meanwhile, our method develops a multi-scale attention module to aggregate node embeddings in higher-order neighborhoods. We evaluate the proposed model with node classification tasks on six datasets: DBLP, ACM, IBDM, Reuters, STUD-BJUT and, Citeseer. Experiments on a large number of benchmarks show the advantages of HGTN.

![image](https://user-images.githubusercontent.com/65967004/193492006-31899ff4-732c-4dc3-8f4b-3592c2c77d8e.png)
### Our paper has been accepted by [ACM TKDD](https://dl.acm.org/doi/10.1145/3565028).
### Our data can be accessed here:
链接：https://pan.baidu.com/s/1jlOCHOQLM0Vp0Ei_DLdYzw
提取码：4j43 
## How to use
> mkdir data 
> 
> python main.py --dataset [ACM/DBLP/IMDB/CITE/REUT/SDUT]

### If you find our work useful in your research, please consider citing:

>  @article{li2023hypergraph,
>   title={Hypergraph transformer neural networks},
>   author={Li, Mengran and Zhang, Yong and Li, Xiaoyong and Zhang, Yuchen and Yin, Baocai},
>   journal={ACM Transactions on Knowledge Discovery from Data},
>   volume={17},
>   number={5},
>   pages={1--22},
>   year={2023},
>   publisher={ACM New York, NY}
> }
