from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from torch.fx.node import Node

from colossalai.logging import get_dist_logger

NON_COMPUTE_OP = ["placeholder", "get_attr", "output"]
NON_COMPUTE_NAME = ["getattr", "eq", "_assert_is_none", "_assert", "finfo", "size"]
logger = get_dist_logger()


def get_logger() -> Any:
    return logger


def flat_list(inputs: Any) -> List:
    """
    flat a list by recursion
    """
    if not (isinstance(inputs, list) or isinstance(inputs, set) or isinstance(inputs, tuple)):
        return [inputs]
    res = []
    for i in inputs:
        if isinstance(i, list) or isinstance(i, set) or isinstance(i, tuple):
            res.extend(flat_list(i))
        else:
            res.append(i)
    return res


def find_first_tensor_arg(node: Node) -> Node:
    """
    Find the first input tensor arg for a node
    """
    for arg in node.args:
        if type(arg) == type(node):
            return arg
    raise RuntimeError()


def is_non_compute_node(node: Node) -> bool:
    if any(i == node.op for i in NON_COMPUTE_OP) or any(i == get_node_name(node) for i in NON_COMPUTE_NAME):
        return True
    if "getitem" in node.name:
        node_args = flat_list(node.args[1:])
        for node_arg in node_args:
            if any(i == str(node_arg) for i in ["None", "Ellipsis"]):
                return False
            if "slice" in str(node_arg):
                return False
        return True
    return False


def get_node_shape(node: Node) -> List:
    if hasattr(node.meta["tensor_meta"], "shape"):
        return node.meta["tensor_meta"].shape
    return None


def is_non_memory_node(node: Node) -> bool:
    if "getitem" in node.name:
        return True
    if "output" in node.op:
        return True
    return is_non_compute_node(node)


def is_non_compute_node_except_placeholder(node: Node) -> bool:
    if "placeholder" in node.op:
        return False
    return is_non_compute_node(node)


def is_non_compute_node_except_placeholder_output(node: Node) -> bool:
    if "output" in node.op:
        return False
    return is_non_compute_node_except_placeholder(node)


def find_idx_by_name(name: str, nodes_list: List) -> int:
    for idx, node in enumerate(nodes_list):
        if node.name == name:
            return idx
    raise RuntimeError("name %s not found in node list" % name)


def delete_free_var_from_last_use(user_to_last_uses: Dict) -> None:
    for key, value in user_to_last_uses.items():
        for n in value:
            if n.op == "placeholder":
                user_to_last_uses[key].remove(n)


def find_chunk_all_input_nodes(nodes: List[Node]) -> List:
    """
    Find non-compute input and output node names.
    input nodes are nodes used in the list
    output nodes are nodes will use nodes in the list
    """
    input_nodes = []
    for node in nodes:
        for input_node in node._input_nodes.keys():
            if input_node not in nodes and input_node not in input_nodes:
                input_nodes.append(input_node)
    return input_nodes


def find_chunk_compute_input_and_output_nodes(nodes: List[Node]) -> Union[List, List]:
    """
    Find non-compute input and output node names.
    input nodes are nodes used in the list
    output nodes are nodes will use nodes in the list
    """
    input_nodes = []
    output_nodes = []

    # if a node has an input node which is not in the node list
    # we treat that input node as the input of the checkpoint function
    for node in nodes:
        for input_node in node._input_nodes.keys():
            if (input_node not in nodes and input_node not in input_nodes
                    and not is_non_compute_node_except_placeholder(input_node)):
                input_nodes.append(input_node)

    # if a node has a user node which is not in the node list
    # we treat that user node as the node receiving the current node output
    for node in nodes:
        for output_node in node.users.keys():
            if (output_node not in nodes and node not in output_nodes
                    and not is_non_compute_node_except_placeholder_output(output_node)):
                output_nodes.append(node)

    return input_nodes, output_nodes


def get_module_node_name(node: Node) -> str:
    """
    get module class name
    """
    node_targets = node.target.split(".")
    module = node.graph.owning_module
    for i in node_targets:
        module = getattr(module, i)
    module_name = str(module.__class__).split(".")[-1][:-2]
    module_name = module_name.lower()
    return module_name


def get_node_name(node: Node) -> str:
    """
    get node name
    """
    node_name = node.name
    if "_" in node_name:
        for i in range(len(node_name) - 1, -1, -1):
            if node_name[i] == "_":
                node_name = node_name[:i]
                break
            elif node_name[i] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                continue
            else:
                break
    return node_name
