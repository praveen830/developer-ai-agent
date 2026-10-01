package com.agent.springbootapi.model;

import java.util.List;

public class Student {

    private Long id;
    private String name;
    private String careerGoal;
    private List<String> skills;

    // Default constructor for serialization/deserialization
    public Student() {
    }

    // Parameterized constructor
    public Student(Long id, String name, String careerGoal, List<String> skills) {
        this.id = id;
        this.name = name;
        this.careerGoal = careerGoal;
        this.skills = skills;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCareerGoal() {
        return careerGoal;
    }

    public void setCareerGoal(String careerGoal) {
        this.careerGoal = careerGoal;
    }

    public List<String> getSkills() {
        return skills;
    }

    public void setSkills(List<String> skills) {
        this.skills = skills;
    }
}
