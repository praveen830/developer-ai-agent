package com.agent.springbootapi.controller;

import com.agent.springbootapi.model.Student;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/api/students")
public class StudentController {

    // In-memory student store for learning/demo purposes
    private final Map<Long, Student> studentStore = new ConcurrentHashMap<>();

    public StudentController() {
        // Initialize student with id 101
        studentStore.put(101L, new Student(
                101L,
                "Praveen",
                "Java Full Stack Developer",
                Arrays.asList("Java", "Spring Boot", "Git")
        ));
    }

    @GetMapping("/{id}")
    public ResponseEntity<?> getStudentById(@PathVariable Long id) {
        Student student = studentStore.get(id);

        if (student != null) {
            return ResponseEntity.ok(student);
        } else {
            Map<String, Object> errorResponse = new LinkedHashMap<>();
            errorResponse.put("status", HttpStatus.NOT_FOUND.value());
            errorResponse.put("error", "Not Found");
            errorResponse.put("message", "Student not found with ID: " + id);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(errorResponse);
        }
    }
}
