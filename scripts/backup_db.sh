run_type=${run_type:-"dev"}
name=${name:-"development"}
user=${user:-"admin"}

docker exec -i $run_type-db pg_dump -W -U $user $name | gzip > db_dump_$(date "+%F-%T").gz