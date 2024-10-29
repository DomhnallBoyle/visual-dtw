# generate self-signed certificate for SSL
common_name=${common_name:-localhost}

openssl req -subj '/CN='$common_name -x509 -nodes -days 365 -newkey rsa:2048 -keyout config/nginx/key.pem -out config/nginx/cert.pem