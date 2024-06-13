import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyTigerGraph as tg
import threading

# TigerGraph connection and authentication
hostName = "https://eaa8afeab8334e1b9f6d6b95b6216f59.i.tgcloud.io"
graphName = "VWG"
secret = "n0r4908rshcutm9vda0qb8c77iep27j0"
userName = "user_1"
password = "A1z2e3r4*"

graph = tg.TigerGraphConnection(host=hostName, graphname=graphName)
authToken = graph.getToken(secret)
authToken = authToken[0]
print(f"secret token: {authToken}")
conn = tg.TigerGraphConnection(host=hostName, graphname=graphName, username=userName, password=password, apiToken=authToken)

# Helper functions
def path_taken(results):
    edges = results[2]['@@display_edge_set']
    node_initial = None
    edge_to_ids = {edge['to_id'] for edge in edges}
    for edge in edges:
        if edge['from_id'] not in edge_to_ids:
            node_initial = edge['from_id']
            break
    if node_initial is None:
        raise ValueError("initial node cannot be determined.")
    current_node = node_initial
    path = [current_node]
    while True:
        next_node = None
        for edge in edges:
            if edge['from_id'] == current_node:
                next_node = edge['to_id']
                break
        if next_node is None:
            break
        path.append(next_node)
        current_node = next_node
    nodes = {node_name: node_name for node_name in path}
    ordered_nodes = [nodes[node_name] for node_name in path]
    return ordered_nodes

def add_or_update_final_nodes_batch(nodes_batch_full, nodes_batch , charge):
    existing_cities = [city for city, _ in nodes_batch_full]
    for node, _ in nodes_batch:
        if node in existing_cities:
            # search if town already in nodes_batch_full
            for i, (city, stock) in enumerate(nodes_batch_full):
                if city == node:
                    # update stock
                    if i == 0:
                        nodes_batch_full[i] = (city, {"Stock": nodes_batch_full[i][1]["Stock"] - charge})
                    else:
                        nodes_batch_full[i] = (city, {"Stock": nodes_batch_full[i][1]["Stock"] + charge})
            continue
        else:
            # Ajout de nouvelles entrées dans nodes_batch_full
            nodes_batch_full.extend([
                (nodes_batch[0][0], {"Stock": nodes_batch[0][1]["Stock"]}),
                (nodes_batch[1][0], {"Stock": nodes_batch[1][1]["Stock"]})
            ])
            return nodes_batch_full
    return nodes_batch_full

def add_or_update_final_edges_batch(edges_batch_full, edges_batch, charge):
    existing_edges = [(edge[0], edge[1]) for edge in edges_batch_full]
    
    for edge in edges_batch:
        # Vérifier si l'arête ou son inverse existe déjà
        if (edge[0], edge[1]) in existing_edges or (edge[1], edge[0]) in existing_edges:
            # Si l'arête existe dans l'une ou l'autre direction
            for i, (origin, destination, movement) in enumerate(edges_batch_full):
                if (origin, destination) == (edge[0], edge[1]) or (origin, destination) == (edge[1], edge[0]):
                    # Mettre à jour le mouvement pour l'arête existante
                    edges_batch_full[i] = (origin, destination, {"Daily_movement": edges_batch_full[i][2]["Daily_movement"] + charge})
            continue
        else:
            # Ajouter de nouvelles entrées à edges_batch_full
            edges_batch_full.append((edge[0], edge[1], {"Daily_movement": edge[2]["Daily_movement"]}))
    
    return edges_batch_full

# Main transfer function
def transfert_nodes_and_edges(node_initial, node_final, charge, weight_attribute="Capacity", nodetype="Nodes", edgetype="distribute_to"):
    global operation_count, palets_number, palets_cost, unique_origin_nodes, unique_destination_nodes, log_list, nodes_batch_full
    results = conn.runInstalledQuery("tg_astar_test", params={
        "source_vertex": node_final, "source_vertex.type": nodetype,
        "target_vertex": node_initial, "target_vertex.type": nodetype,
        "e_type_set": edgetype, "weight_type": "FLOAT",
        "latitude": "latitude", "longitude": "longitude",
        "weight_attribute": weight_attribute,
        "print_stats": "True"
    })

    order_taken = path_taken(results)
    edge_set = results[2]['@@display_edge_set']
    reordered_edges_set = []
    for location in order_taken:
        for edge in edge_set:
            if edge['from_id'] == location:
                reordered_edges_set.append(edge)

    node_set = results[2]['tmp']
    reordered_nodes_set = []
    for location in order_taken:
        for item in node_set:
            if item['v_id'] == location:
                reordered_nodes_set.append(item)

    try:
        node_i_stock = reordered_nodes_set[0]["attributes"]["Stock"]
        node_f_stock = reordered_nodes_set[-1]["attributes"]["Stock"]
        node_i_unload_capacity = reordered_nodes_set[0]["attributes"]["UnloadCapacity"]
        node_f_load_capacity = reordered_nodes_set[-1]["attributes"]["LoadCapacity"]

        if charge > node_i_stock:
            error_message = f"{node_initial} to {node_final}: Not enough stock ({charge} > {node_i_stock})"
            log_list.append(error_message)

        if charge > node_f_load_capacity:
            error_message = f"{node_initial} to {node_final}: Not enough load capacity from receiving warehouse (charge: {charge} > LoadCapacity: {node_f_load_capacity})"
            log_list.append(error_message)

        if charge > node_i_unload_capacity:
            error_message = f"{node_initial} to {node_final}: Not enough unload capacity from starting warehouse (charge: {charge} > UnloadCapacity: {node_i_unload_capacity})"
            log_list.append(error_message)

        capacity_edge = reordered_edges_set[0]["attributes"]["Capacity"]
        daily_movement = reordered_edges_set[0]["attributes"]["Daily_movement"]

        if daily_movement + charge > capacity_edge:
            error_message = f"Send {capacity_edge} from {node_initial} to {node_final}: Not enough edge capacity ({daily_movement + charge} > {capacity_edge})"
            log_list.append(error_message)

        nodes_batch = [
            (node_initial, {"Stock": node_i_stock - charge}),
            (node_final, {"Stock": node_f_stock + charge})
        ]

        edges_batch = []
        for element in reordered_edges_set:
            element['attributes']['Daily_movement'] += charge
            edges_batch.append((element["from_id"], element["to_id"], {"Daily_movement": element["attributes"]["Daily_movement"]}))
            palets_cost += element["attributes"]["Price"]

        unique_origin_nodes.add(node_initial)
        unique_destination_nodes.add(node_final)
        operation_count += 1
        palets_number += charge

        return nodes_batch , edges_batch

    except Exception as e:
        error_message = f"Error in the transfer: {e}"
        log_list.append(error_message)
        return error_message

# Thread safety
lock = threading.Lock()

def make_daily_movements(input_file, column_origin, column_destination, column_transfert):
    global operation_count, palets_number, palets_cost, unique_origin_nodes, unique_destination_nodes, log_list, nodes_batch_full
    operation_count = 0
    palets_number = 0
    palets_cost = 0
    unique_origin_nodes = set()
    unique_destination_nodes = set()
    log_list = []
    nodes_batch_full = []
    edges_batch_full = []

    df = pd.read_csv(input_file)

    def perform_transfer(row):
        try:
            origin = str(row[column_origin])
            destination = str(row[column_destination])
            volume = float(row[column_transfert])
            nodes_batch, edges_batch= transfert_nodes_and_edges(origin, destination, volume)
            with lock:
                add_or_update_final_nodes_batch(nodes_batch_full, nodes_batch, volume)
                add_or_update_final_edges_batch(edges_batch_full, edges_batch, volume )
        except Exception as e:
            log_list.append(f"Error in make_daily_movements: {e}")

    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = [executor.submit(perform_transfer, row) for index, row in df.iterrows()]
        for future in as_completed(futures):
            pass

    conn.upsertVertices("Nodes", nodes_batch_full)
    nodetype = "Nodes"
    edgetype = "distribute_to"

    conn.upsertEdges(sourceVertexType=nodetype, targetVertexType=nodetype, edgeType=edgetype, edges=edges_batch_full)

    return log_list

log_messages = make_daily_movements(r"C:\Users\JulienRigot\OneDrive - LIS Data Solutions\Escritorio\code_GORDIAS\base de datos graph\Tigergraph\df_light_1000.csv", "CODE_Origin", "CODE_Destination", "Palets")
print(log_messages)