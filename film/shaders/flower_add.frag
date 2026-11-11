#version 410
in vec2 uv; out vec4 o;
uniform sampler2D uTex;
uniform float uW;
void main(){ o=texture(uTex,uv)*uW; }
