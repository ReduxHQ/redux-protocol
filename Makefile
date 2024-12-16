

server_ip := 178.156.153.83  # your server ip
git_email := $(GIT_EMAIL)
git_name := $(GIT_NAME)

help:
	@echo "make copy"
	@echo "make rebuild"
	@echo "make restart-caddy"
	@echo "make restart-server"

# copy to server
copy: # copy to server
	./scripts/copy.sh
	# copy env file to server
	scp .env.production root@178.156.153.83:/app/.env


rebuild: # rebuild docker image
	./scripts/rebuild.sh

build_es: # build eliza-server
	cd packages/core/ && pnpm install && pnpm run build
	cd packages/client-twitter/ && pnpm install && pnpm run build
	cd packages/client-direct/ && pnpm install && pnpm run build


add-git-config: # add git config
	git config --global user.email $(git_email)
	git config --global user.name $(git_name)

test-ssh: # test ssh
	ssh -T git@github.com

push: # push to remote
	git push -u redux develop

copy-to-server: # copy to server
	./scripts/copy.sh

restart-caddy: # restart caddy
	ssh root@$(server_ip) "cd /app && caddy reload"

restart-server: # restart server
	ssh root@$(server_ip) "cd /app && make rebuild"
