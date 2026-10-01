# agent-springboot-api

A minimal standalone Spring Boot REST API application exposing a student profile endpoint for learning and demo purposes.

---

## 1. What the Application Does
This application is a lightweight Spring Boot microservice designed to demonstrate building a clean REST API without external database or JPA overhead. It stores student profile records in-memory and provides a REST endpoint to query a student by their ID.

- **Pre-loaded Student ID**: `101` (Praveen)
- **Unknown Student IDs**: Handled with an HTTP `404 Not Found` JSON error response.

---

## 2. Technology Stack
- **Language**: Java 17
- **Framework**: Spring Boot 3.3.4
- **Dependency**: `spring-boot-starter-web`
- **Build Tool**: Apache Maven

---

## 3. How to Run

### Prerequisites
- JDK 17 or higher installed (`java -version`)
- Apache Maven installed (`mvn -version`)

### Build the Project
Open a terminal in the `agent-springboot-api` folder and compile:
```bash
mvn clean package
```

### Start the Application
Run using the Spring Boot Maven plugin:
```bash
mvn spring-boot:run
```
Alternatively, run the packaged JAR directly:
```bash
java -jar target/agent-springboot-api-0.0.1-SNAPSHOT.jar
```

The application will start on port `8080` by default.

---

## 4. Endpoints & Example Responses

### Endpoint 1: Retrieve Student Profile (Success)
- **Method**: `GET`
- **URL**: `http://localhost:8080/api/students/101`
- **HTTP Status**: `200 OK`
- **Response Body**:
```json
{
  "id": 101,
  "name": "Praveen",
  "careerGoal": "Java Full Stack Developer",
  "skills": [
    "Java",
    "Spring Boot",
    "Git"
  ]
}
```

### Endpoint 2: Retrieve Unknown Student Profile (Not Found)
- **Method**: `GET`
- **URL**: `http://localhost:8080/api/students/999`
- **HTTP Status**: `404 Not Found`
- **Response Body**:
```json
{
  "status": 404,
  "error": "Not Found",
  "message": "Student not found with ID: 999"
}
```
