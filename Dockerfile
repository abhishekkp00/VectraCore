# Build stage
FROM eclipse-temurin:17-jdk-alpine AS builder
WORKDIR /app
COPY src/ ./src/
RUN mkdir bin && javac -d bin src/*.java

# Run stage
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=builder /app/bin ./bin
COPY web/ ./web/
EXPOSE 8080
CMD ["java", "-cp", "bin", "Main"]
