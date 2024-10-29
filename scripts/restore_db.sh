gzip_path=${gzip_path:-""}
name=${name:-"development"}
user=${user:-"admin"}
host=${host:-"localhost"}
port=${port:-5432}

gunzip -c $gzip_path | psql -d $name -U $user -h $host -p $port