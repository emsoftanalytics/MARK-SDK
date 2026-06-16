FROM python:3.11-alpine

RUN addgroup -S sandbox && adduser -S sandbox -G sandbox

WORKDIR /sandbox
RUN mkdir -p /sandbox/workspace \
    && chown -R sandbox:sandbox /sandbox

USER sandbox
CMD ["python", "/sandbox/code.py"]
