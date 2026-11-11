#version 410
// Back out of the full stop: the frame of the small world shrinks around its sun,
// and the sentence comes in around it. Output is display-referred, like the small world.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uBg,uFg;
uniform vec2 uSun,uPos;
uniform float uS,uEV;
vec3 tone(vec3 x){
  x=max(x,0.);
  vec3 y=(x*(2.51*x+.03))/(x*(2.43*x+.59)+.14);
  return clamp(y,0.,1.);
}
void main(){
  vec2 q=uSun+(uv-uPos)/uS;
  vec3 f=vec3(0.);
  if(q.x>0.&&q.x<1.&&q.y>0.&&q.y<1.) f=texture(uFg,q).rgb;
  vec3 b=lin2srgb(tone(texture(uBg,uv).rgb*pow(2.,uEV)));
  o=vec4(max(b,f),1.);
}
