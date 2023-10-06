(RAYenv) [mamo@gpu1-1g ray_tmp]$ python collective_communication_barrier_example.py
2023-10-06 13:35:43,343 INFO worker.py:1633 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265
Ray initialized.
Workers created.
Starting barrier operation across workers...
(Worker pid=89242) (Worker 0) Collective group initialized.
(Worker pid=89242) (Worker 0) Before barrier.
2023-10-06 13:35:46,719 WARNING worker.py:2058 -- It looks like you're creating a detached actor in an anonymous namespace. In order to access this actor in the future, you will need to explicitly connect to this namespace with ray.init(namespace="c8cb3e03-5cbe-4c7a-840a-6c0639eecdc1", ...)
(Worker pid=89243) (Worker 1) Collective group initialized.
(Worker pid=89243) (Worker 1) Before barrier.

