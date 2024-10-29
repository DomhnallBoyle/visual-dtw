env=${env:-"dev"}

# add docker environment variables at run time to a file in root folder
# to be used by the cronjob shell processes
printenv > /container.env

# edit crontab to include new jobs
crontab config/cron/${env}_cronjobs

# run jobs in background
cron