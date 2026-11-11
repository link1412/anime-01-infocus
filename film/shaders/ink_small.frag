#version 410
// A small point-sampled copy of the frame: every texel is a real pixel of the picture, not an average.
in vec2 uv; out vec4 o;
uniform sampler2D uTex;
void main(){ o=texelFetch(uTex,ivec2(gl_FragCoord.xy*8.),0); }
