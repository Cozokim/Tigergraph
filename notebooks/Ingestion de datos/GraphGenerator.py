from typing import Dict
from create_loading_jobs_template_vwg import create_loading_jobs_template_vwg
from schema_vwg import schema_vwg
import pyTigerGraph as tg

class GraphGenerator:
    def __init__(self, conn, schema: Dict, data_source_dict: Dict):
        # Initialize with connection object, schema, and data sources for vertices and edges.


#         Example of data_source_dict : 
#         DEFAULT_DATA_SOURCE_DICT = {
#                        "vertex_data_job": r"Tigergraph\data\Europe\biggest_europe_citie.csv",
#                        "edge_data_job": r"Tigergraph\data\Europe\conexiones_biggest_europe_cities.csv"
#                        }
#       with vertex_data and edge_data being the name of the loading job

        self.conn = conn
        self.schema = schema
        self.data_source_dict = data_source_dict
        self.graph_name = schema['GraphName']

    def create_loading_jobs(self):
        # Create loading jobs using the template, substituting the graph name.
        try:
            loading_jobs = create_loading_jobs_template_vwg.substitute({
                'graph_name': self.graph_name,
            })
            print(self.conn.gsql(loading_jobs))
        except Exception as e:
            print(f"Error creating loading jobs: {str(e)}")

    def load_from_files(self):
        # Execute loading jobs using the specified data sources.
        created_jobs = self.conn.gsql(f"USE GRAPH {self.graph_name} SHOW JOB *")
        jobs_to_rerun = []

        # Load data from files for each job
        for job_name, file_path in self.data_source_dict.items():
            
            if job_name in created_jobs:
                try:
                    print(f"---- Uploading file {file_path} for job {job_name} ---- \n")
                    response = self.conn.runLoadingJobWithFile(
                        filePath=file_path,
                        fileTag="MyDataSource",
                        jobName=job_name,
                        timeout=32000
                    )
                    print(response)
                except Exception as e:
                    print(f"Job {job_name} failed: {str(e)}")
                    jobs_to_rerun.append(job_name)
            else:
                print(f"Job {job_name} does not exist.")
        
        print(f"Jobs that failed: {jobs_to_rerun}")
        
    def execute_pipeline(self):
        # Execute the full pipeline: loading job creation, and data loading.
        self.create_loading_jobs()
        self.load_from_files()
