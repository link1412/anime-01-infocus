#version 410
// what the light meets, first and second: the sheet is thin, so the light goes on, changed
in vec2 vUv; in float vL; in vec3 vP;
out vec4 o;
uniform sampler2D uPig,uZ1;
uniform float uPeel,uBias;
uniform vec2 uTileO,uTileS;
void main(){
  if(texture(uPig,vUv).a<.5) discard;
  if(uPeel>.5){
    float z1=texelFetch(uZ1,ivec2(gl_FragCoord.xy),0).r;
    if(vL<=z1+uBias) discard;
  }
  o=vec4(vL,0.,0.,1.);
}
