#version 300 es

precision highp float;
// uniform sampler2D u_Texture;
uniform vec4 color;
uniform vec4 border_color;

in vec2 v_TexCoord;
layout(location = 0) out vec4 o_FragColor;

void main() {
  // o_FragColor = texture(u_Texture, v_TexCoord);

  o_FragColor = (v_TexCoord.x < 0.02) ? border_color
                : (v_TexCoord.y < 0.02) ? border_color
                : (v_TexCoord.x > 0.98) ? border_color
                : (v_TexCoord.y > 0.98) ? border_color
                : color;
}
