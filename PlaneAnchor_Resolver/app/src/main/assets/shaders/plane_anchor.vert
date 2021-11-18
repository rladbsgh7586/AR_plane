#version 300 es

//uniform mat4 u_Model;
uniform mat4 u_ModelViewProjection;
//uniform mat2 u_PlaneUvMatrix;
//uniform vec3 u_Normal;

layout(location = 0) in vec3 vertex; // (x, y, z)
layout(location = 1) in vec2 vertex_uv; // (u, v)

out vec2 v_TexCoord;

void main() {
   vec4 local_pos = vec4(vertex, 1.0);

   v_TexCoord = vertex_uv;

   gl_Position = u_ModelViewProjection * local_pos;
}