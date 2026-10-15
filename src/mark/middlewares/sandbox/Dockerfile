FROM python:3.11-alpine

RUN addgroup -S sandbox && adduser -S sandbox -G sandbox

WORKDIR /sandbox
RUN mkdir -p /sandbox/workspace \
    && chown -R sandbox:sandbox /sandbox

USER sandbox
# Code is streamed over stdin by DockerSandbox (`docker run -i ... python -`).
CMD ["python", "-"]
