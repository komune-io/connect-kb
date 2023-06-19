KB_DOCKERFILE	:= infra/Dockerfile
KB_NAME	    	:= smartbcity/kb
KB_IMG	    	:= ${KB_NAME}:${VERSION}
KB_LATEST		:= ${KB_NAME}:latest

docker: docker-push

docker-push:
	@docker build -f ${KB_DOCKERFILE} -t ${KB_IMG} .
	@docker push ${KB_IMG}

docker-push-latest:
	@docker build -f ${KB_DOCKERFILE} -t ${KB_LATEST} .
	@docker push ${KB_LATEST}
