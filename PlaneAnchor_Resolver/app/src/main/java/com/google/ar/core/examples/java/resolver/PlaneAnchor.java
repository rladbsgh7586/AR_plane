package com.google.ar.core.examples.java.resolver;

public class PlaneAnchor {
    float[] transformation_matrix;
    String plane_name;
    int width;
    int height;

    PlaneAnchor(float[] A, int B, int C, String D){
        this.transformation_matrix = A.clone();
        this.width = B;
        this.height = C;
        this.plane_name = D;
    }

    PlaneAnchor(){

    }

    public void setTransformationMatrix(float[] A){
        this.transformation_matrix = A;
    }

    public void setPlaneName(String A){
        this.plane_name = A;
    }

    public float[] getTransformationMatrix(){
        return this.transformation_matrix;
    }

    public String getName(){
        return this.plane_name;
    }

    public int getWidth(){
        return this.width;
    }

    public int getHeight(){
        return this.height;
    }
}
