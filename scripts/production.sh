# script to start the production environment with sensitive credentials
env=${env:-"prod"}
make_type=${make_type:-"start_background"}

read -s -p "Enter database username: " POSTGRES_USER && echo
read -s -p "Enter database password: " POSTGRES_PASSWORD && echo

export POSTGRES_USER=${POSTGRES_USER:-"admin"}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-"password"}

make $make_type ENV=$env