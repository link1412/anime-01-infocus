#version 410
// Prepare the diffusion state: the sharp painting with this frame's fibre conductivity in alpha.
// Conductivity is anchored to the paper, so it moves with the pan and never swims.
in vec2 uv; out vec4 o;
uniform sampler2D uD,uK;
uniform float uAspect,uZoom,uPhPaper,uPaperP;
uniform vec2 uZc;
void main(){
  vec2 p=vec2((uv.x*2.-1.)*uAspect,uv.y*2.-1.);
  vec2 v=uZc*(1.-1./uZoom)+p/uZoom;
  vec2 X=vec2(v.x+uPhPaper,v.y);
  float k=texture(uK,X/uPaperP).r;
  // when magnified, finer strands of the same paper take over the conduction
  k=mix(k,texture(uK,X*7.13/uPaperP+.37).r,smoothstep(2.,9.,uZoom));
  o=vec4(texture(uD,uv).rgb,k);
}
