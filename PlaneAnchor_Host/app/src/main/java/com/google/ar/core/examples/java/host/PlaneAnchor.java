package com.google.ar.core.examples.java.host;

public class PlaneAnchor {
    float[] transformation_matrix;
    int width;
    int height;

    PlaneAnchor(float[] A, int B, int C){
        this.transformation_matrix = A.clone();
        this.width = B;
        this.height = C;
    }

    PlaneAnchor(){

    }

    public void setTransformationMatrix(float[] A){
        this.transformation_matrix = A;
    }

    public float[] getTransformationMatrix(){
        return this.transformation_matrix;
    }

    public int getWidth(){
        return this.width;
    }

    public int getHeight(){
        return this.height;
    }
}
