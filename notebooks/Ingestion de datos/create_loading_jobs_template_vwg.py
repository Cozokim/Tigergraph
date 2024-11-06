from string import Template

# This file defines the template for creating loading jobs for the VWG graph.
create_loading_jobs_template_vwg = Template("""
USE GRAPH $graph_name

CREATE LOADING JOB vertex_data_job FOR GRAPH $graph_name {
    DEFINE FILENAME MyDataSource;
    LOAD MyDataSource TO VERTEX Nodes VALUES($$0, $$1, $$2, $$3, $$4, $$5, $$6, $$7) USING SEPARATOR=",", HEADER="true", EOL="\\n";
}
set exit_on_error = "true"
set exit_on_error = "false"

CREATE LOADING JOB edge_data_job FOR GRAPH $graph_name {
    DEFINE FILENAME MyDataSource;
    LOAD MyDataSource TO EDGE distribute_to VALUES($$0, $$1, $$2, $$3, $$4 , $$5) USING SEPARATOR=",", HEADER="true", EOL="\\n";
}
set exit_on_error = "true"
set exit_on_error = "false"
""")
