#version 410
in vec2 uv; out vec4 o; uniform sampler2D uTex;
void main(){ o=vec4(texture(uTex,uv).rgb,1.); }
