(RAYenv) [mamo@gpu1-1g ray_tmp]$ python collective_communication_broadcast_example_output.py
2023-10-06 10:45:36,447 INFO worker.py:1633 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265
(Worker pid=62623) Rank 0: Initialized data with its rank value.
(Worker pid=62623) Rank 0: Initialized collective group.
(Worker pid=62624) Rank 1: Initialized data with zeros.
2023-10-06 10:45:39,694 WARNING worker.py:2058 -- It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init(namespace="27ca2072-9821-4e96-aaf2-dd6bd31c0177", ...)
(Worker pid=62623) Rank 0: Completed broadcast operation.
Rank 0 data: [0. 0. 0. 0.]
Rank 1 data: [0. 0. 0. 0.]
(Worker pid=62624) Rank 1: Initialized collective group.
(Worker pid=62624) Rank 1: Completed broadcast operation.
(RAYenv) [mamo@gpu1-1g ray_tmp]$

