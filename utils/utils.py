import os
from datetime import datetime
from torch_geometric.utils.convert import to_networkx
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.components import connected_components

def printParOnFile(test_name, log_dir, par_list):
    '''
    function that initialize a test log file
    :param test_name: test name
    :param log_dir: where the file will be saved
    :param par_list: dictionary that contains the hyper-parameters of the model
    :return:
    '''

    assert isinstance(par_list, dict), "par_list as to be a dictionary"
    f=open(os.path.join(log_dir,test_name+".log"),'w+')
    f.write(test_name)
    f.write("\n")
    f.write(str(datetime.now().utcnow()))
    f.write("\n\n")
    for key, value in par_list.items():
        f.write(str(key)+": \t"+str(value))
        f.write("\n")


def get_graph_diameter(data):
    '''
    compute the graph diameter and add the attribute to data object
    :param data: the graph
    :return: the graph representation augmented with diameter attribute
    '''
    networkx_graph = to_networkx(data).to_undirected()

    sub_graph_list = [networkx_graph.subgraph(c) for c in connected_components(networkx_graph)]
    sub_graph_diam = []
    for sub_g in sub_graph_list:
        sub_graph_diam.append(diameter(sub_g))
    data.diameter=max(sub_graph_diam)
    return data
