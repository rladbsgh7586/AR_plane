#version 300 es

precision highp float;
uniform sampler2D u_Texture;

in vec2 v_TexCoord;
layout(location = 0) out vec4 o_FragColor;

void main() {
  o_FragColor = texture(u_Texture, v_TexCoord);
}
