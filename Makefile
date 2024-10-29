ifeq ($(filter $(ENV),test dev prod_single prod_distributed research),)
    $(error The ENV variable is invalid (dev, test, prod_single, prod_distributed, research).)
endif

$(shell cat docker-compose.anchors.yml docker-compose.$(ENV).yml > docker-compose.config.yml)

ifneq ($(filter $(ENV),test dev research),)
    OVERRIDE_CMD = $(shell echo -f docker-compose.common.yml -f docker-compose.config.yml)
else
    OVERRIDE_CMD = $(shell echo -f docker-compose.common.yml -f docker-compose.prod.yml -f docker-compose.config.yml)
endif

show_config:
	$(info Make: Building configuration for "$(ENV)".)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) config

build:
	$(info Make: Building "$(ENV)" environment images.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) build

build_service:
	$(info Make: Building "$(ENV)" environment images.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) build $(SERVICE)

build_start:
	$(info Make: Building and starting "$(ENV)" environment containers.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) up --build

build_start_background:
	$(info Make: Building and starting "$(ENV)" environment containers in background.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) up --build -d

build_service_start_background:
	$(info Make: Building and starting "$(ENV)" environment service $(SERVICE) in background.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) up -d --build $(SERVICE)

start:
	$(info Make: Starting "$(ENV)" environment containers.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) up

start_background:
	$(info Make: Starting "$(ENV)" environment containers in background.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) up -d

start_service:
	$(info Make: Starting "$(ENV)" environment service $(SERVICE).)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) up $(SERVICE)

stop:
	$(info Make: Stopping "$(ENV)" environment containers.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) stop

down:
	$(info Make: Removing "$(ENV)" environment containers and volumes.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) down --volumes

restart:
	$(info Make: Refreshing "$(ENV)" environment containers.)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) restart

clean:
	$(info Make: Cleaning "$(ENV)" environment (removing containers, images & volumes).)
	@docker-compose -p $(ENV) $(OVERRIDE_CMD) down --rmi all --volumes

clean_all:
	$(info Make: Cleaning all environments (removing containers, images & volumes).)
	@make -s clean ENV=test
	@make -s clean ENV=dev
	@make -s clean ENV=prod
