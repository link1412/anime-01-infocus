#version 410
// For the lens: the largest blur that can reach each tile of the image (r: anything; g: what is in front of the glass).
in vec2 uv; out vec4 o;
uniform sampler2D uRoom,uPrev;
uniform vec2 uRoomSize;
uniform float uKc,uFd,uIdG,uDilate;
void main(){
  if(uDilate>.5){
    vec2 m=vec2(0.);
    ivec2 ip=ivec2(gl_FragCoord.xy);
    ivec2 sz=textureSize(uPrev,0);
    for(int j=-8;j<=8;j++) for(int i=-8;i<=8;i++){
      ivec2 q=clamp(ip+ivec2(i,j),ivec2(0),sz-1);
      m=max(m,texelFetch(uPrev,q,0).rg);
    }
    o=vec4(m,0.,1.);
    return;
  }
  ivec2 c=ivec2(gl_FragCoord.xy)*16;
  float a=0.,f=0.;
  for(int j=0;j<16;j+=2) for(int i=0;i<16;i+=2){
    float id=texelFetch(uRoom,min(c+ivec2(i,j),ivec2(uRoomSize)-1),0).a;
    if(id<=0.) continue;
    float cc=uKc*abs(id-uFd);
    a=max(a,cc);
    if(id>uIdG+.004) f=max(f,max(cc,.8));
  }
  o=vec4(a,f,0.,1.);
}
