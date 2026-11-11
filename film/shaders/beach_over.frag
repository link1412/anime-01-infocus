#version 410
// What the air adds over the finished radiance: blended with ONE, SRC_ALPHA
in vec2 uv; out vec4 o;
uniform sampler2D uTex;
void main(){ o=texture(uTex,uv); }
