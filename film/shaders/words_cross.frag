#version 410
// The last moments of the painting: its characters hand over to the wall. Display-referred, like the painting.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uInk,uWords;
uniform float uK,uEV;
vec3 tone(vec3 x){
  x=max(x,0.);
  vec3 y=(x*(2.51*x+.03))/(x*(2.43*x+.59)+.14);
  return clamp(y,0.,1.);
}
void main(){
  vec3 a=texture(uInk,uv).rgb;
  vec3 b=lin2srgb(tone(texture(uWords,uv).rgb*pow(2.,uEV)));
  o=vec4(mix(a,b,uK),1.);
}
