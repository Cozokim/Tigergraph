# This file contains the schema definition for the VWG graph, which includes the vertex and edge configurations.
schema_vwg = {
    'GraphName': 'VWG',
    'VertexTypes': [
        {
            'Config': {'STATS': 'OUTDEGREE_BY_EDGETYPE'},
            'Attributes': [
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'LoadCapacity'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'UnloadCapacity'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'Stock'},
                {'AttributeType': {'Name': 'STRING'}, 'AttributeName': 'Carga'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'Capacity'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'latitude'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'longitude'}
            ],
            'PrimaryId': {'AttributeType': {'Name': 'STRING'}, 'AttributeName': 'id'},
            'Name': 'Nodes'
        }
    ],
    'EdgeTypes': [
        {
            'IsDirected': False,
            'ToVertexTypeName': 'Nodes',
            'Config': {},
            'Attributes': [
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'Price'},
                {'AttributeType': {'Name': 'STRING'}, 'AttributeName': 'Carga'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'Capacity'},
                {'AttributeType': {'Name': 'FLOAT'}, 'AttributeName': 'Daily_movement'}
            ],
            'FromVertexTypeName': 'Nodes',
            'Name': 'distribute_to'
        }
    ],
    'UDTs': []
}
