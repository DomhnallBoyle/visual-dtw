run_type=${run_type:-"dev"}
name=${name:-"development"}
user=${user:-"admin"}

docker exec -i $run_type-db pg_dump -s -t "PAVA*" -t "config" -t "alembic_version" -d $name -U $user -W | cat > schema.sql